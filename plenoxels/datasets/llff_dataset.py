import glob
import os,imageio
import logging as log
from typing import Tuple, Optional, List

import numpy as np
import torch
import random

from .data_loading import parallel_load_images
from .ray_utils import (
    center_poses, generate_spiral_path, create_meshgrid, stack_camera_dirs, get_rays
)
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset
from pathlib import Path

from .colmapUtils.read_write_model import read_images_binary,read_points3d_binary
from .colmapUtils.read_write_dense import *
from . import colmapUtils

import json

class LLFFDataset(BaseDataset):
    def __init__(self,
                 datadir,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: int = 4,
                 hold_every: int = 8,
                 contraction: bool = False,
                 ndc: bool = True,
                 near_scaling: float = 0.9,
                 ndc_far: float = 1.0,
                 ):
        if (not contraction) and (not ndc):
            raise ValueError("LLFF dataset expects either contraction or NDC to be enabled.")
        self.downsample = downsample
        self.hold_every = hold_every
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far

        if split == 'render':
            # For rendering path we load all poses, use them to generate spiral poses.
            # No gt images exist!
            assert ndc, "Unable to generate render poses without ndc: don't know near-far."
            image_paths, poses, near_fars, intrinsics, self.depth_data = load_llff_poses(
                datadir, downsample=downsample, split='test', hold_every=1,
                near_scaling=self.near_scaling)
            render_poses = generate_spiral_path(
                poses.numpy(), near_fars.numpy(), n_frames=120, n_rots=2, zrate=0.5,
                dt=self.near_scaling)
            self.poses = torch.from_numpy(render_poses).float()
            imgs = None
        else:
            image_paths, self.poses, near_fars, intrinsics, self.depth_data = load_llff_poses(
                datadir, downsample=downsample, split=split, hold_every=hold_every,
                near_scaling=self.near_scaling)
            
            imgs = load_llff_images(image_paths, intrinsics, split)
            imgs = (imgs * 255).to(torch.uint8)
            if split == 'train':
                imgs = imgs.view(-1, imgs.shape[-1])
            else:
                imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])
        num_images = len(self.poses)
        if contraction:
            bbox = torch.tensor([[-2., -2., -2.], [2., 2., 2.]])
            self.near_fars = near_fars
        else:
            bbox = torch.tensor([[-1.5, -1.67, -1.], [1.5, 1.67, 1.]])
            self.near_fars = torch.tensor([[0.0, self.ndc_far]]).repeat(num_images, 1)

        # These are used when contraction=True
        self.global_translation = torch.tensor([0, 0, 1.5])
        self.global_scale = torch.tensor([0.9, 0.9, 1])

        super().__init__(
            datadir=datadir,
            split=split,
            scene_bbox=bbox,
            batch_size=batch_size,
            imgs=imgs,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            is_ndc=ndc,
            is_contracted=contraction,
        )
        log.info(f"LLFFDataset. {contraction=} {ndc=}. Loaded {split} set from {datadir}. "
                 f"{num_images} poses of shape {self.img_h}x{self.img_w}. "
                 f"Images loaded: {imgs is not None}. Near-far[:3]: {self.near_fars[:3]}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")


    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)
            image_id = torch.div(index, h * w, rounding_mode='floor')
            y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
            x = torch.remainder(index, h * w).remainder(w)
            x = x + 0.5
            y = y + 0.5
        else:
            image_id = [index]
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)
        out = {"near_fars": self.near_fars[image_id, :].view(-1, 2)}
        if self.imgs is not None:
            out["imgs"] = self.imgs[index] / 255.0  # (num_rays, 3)   this converts to f32
        else:
            out["imgs"] = None

        c2w = self.poses[image_id]       # (num_rays, 3, 4)
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, True)  # [num_rays, 3]
        rays_o, rays_d = get_rays(camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0,
                                  intrinsics=self.intrinsics, normalize_rd=True)  # h*w, 3

        if self.split == 'train':
            K = 241 # Ideally we would make this configurable
            x = []
            y = []
            depth = []
            weight = []
            image_ids = []

            train_image_ids = torch.unique(image_id)
            for image_id in train_image_ids:
                coords  = self.depth_data[image_id]['coord']  # Shape [N, 2]
                depths  = self.depth_data[image_id]['depth']  # Shape [N]
                weights = self.depth_data[image_id]['error']  # Shape [N]

                N = coords.shape[0]
                k = min(K,N)

                # Randomly select K indices from the range [0, N)
                random_indices = random.sample(range(N), k)

                # Append the sampled coordinates, depths, and image IDs to the lists
                x.extend(coords[random_indices][:, 0].tolist())
                y.extend(coords[random_indices][:, 1].tolist())
                depth.extend(depths[random_indices].tolist())
                weight.extend(weights[random_indices].tolist())
                image_ids.extend([image_id] * k)

            # Convert lists to torch tensors,
            # For now, we need equal dimensions as the rays sampled for the reconstruction loss
            size        = index.size()[0]
            x           = torch.tensor(x[:size])
            y           = torch.tensor(y[:size])
            depth       = torch.tensor(depth[:size])
            weight      = torch.tensor(weight[:size])
            image_ids   = torch.tensor(image_ids[:size])
        else:
            image_ids   = [index]
            x, y        = self.depth_data[image_ids]['coord']
            depth       = self.depth_data[image_ids]['depth']
            weight      = self.depth_data[image_ids]['error']
        out = {"near_fars": self.near_fars[image_ids, :].view(-1, 2)}
        if self.imgs is not None:
            out["imgs"] = self.imgs[index] / 255.0  # (num_rays, 3)   this converts to f32
        else:
            out["imgs"] = None

        c2w = self.poses[image_ids]       # (num_rays, 3, 4)
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, True)  # [num_rays, 3]
        rays_sparse_o, rays_sparse_d = get_rays(camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0,
                                                intrinsics=self.intrinsics, normalize_rd=True)  # h*w, 3
        
        out["rays_o"] = rays_o
        out["rays_d"] = rays_d
        out["rays_sparse_o"] = rays_sparse_o
        out["rays_sparse_d"] = rays_sparse_d
        out["sparse_depth"] = depth
        out["sparse_weight"] = weight
        out["bg_color"] = torch.tensor([[1.0, 1.0, 1.0]])
        return out


def _split_poses_bounds(poses_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    near_fars = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
    intrinsics = Intrinsics(
        width=W, height=H, focal_x=focal, focal_y=focal, center_x=W / 2, center_y=H / 2)
    return poses[:, :, :4], near_fars, intrinsics


def load_llff_poses_helper(datadir: str, downsample: float, near_scaling: float) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses_bounds = np.load(os.path.join(datadir, 'poses_bounds.npy'))  # (N_images, 17)
    poses, near_fars, intrinsics = _split_poses_bounds(poses_bounds)

    # Step 1: rescale focal length according to training resolution
    intrinsics.scale(1 / downsample)

    # Step 2: correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    # (N_images, 3, 4) exclude H, W, focal
    poses, pose_avg = center_poses(poses)

    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    # See https://github.com/bmild/nerf/issues/34
    near_original = np.min(near_fars)
    scale_factor = near_original * near_scaling  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33
    near_fars /= scale_factor
    poses[..., 3] /= scale_factor

    return poses, near_fars, intrinsics


def load_llff_poses(datadir: str,
                    downsample: float,
                    split: str,
                    hold_every: int,
                    near_scaling: float = 0.75) -> Tuple[
                        List[str], torch.Tensor, torch.Tensor, Intrinsics]:
    int_dsample = int(downsample)
    if int_dsample != downsample or int_dsample not in {4, 8}:
        raise ValueError(f"Cannot downsample LLFF dataset by {downsample}.")

    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    image_paths = sorted(glob.glob(os.path.join(datadir, f'images_{int_dsample}/*')))
    assert poses.shape[0] == len(image_paths), \
        'Mismatch between number of images and number of poses! Please rerun COLMAP!'

    depth_data = load_colmap_depth(datadir, downsample, split, near_scaling)

    # Take training or test split
    i_test = np.arange(0, poses.shape[0], hold_every)
    img_list = i_test if split != 'train' else list(set(np.arange(len(poses))) - set(i_test))
    img_list = np.asarray(img_list)

    depth_data = depth_data[img_list]
    image_paths = [image_paths[i] for i in img_list]
    poses = torch.from_numpy(poses[img_list]).float()
    near_fars = torch.from_numpy(near_fars[img_list]).float()

    return image_paths, poses, near_fars, intrinsics, depth_data


def load_llff_images(image_paths: List[str], intrinsics: Intrinsics, split: str):
    all_rgbs: List[torch.Tensor] = parallel_load_images(
        tqdm_title=f'Loading {split} data',
        dset_type='llff',
        data_dir='/',  # paths from glob are absolute
        num_images=len(image_paths),
        paths=image_paths,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
    )
    return torch.stack(all_rgbs, 0)



def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)


def load_colmap_depth(basedir:str, factor:float, split: str, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    poses = get_poses(images)
    _, bds_raw = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = bds_raw.T
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":torch.from_numpy(np.array(depth_list)).float(), "coord":torch.from_numpy(np.array(coord_list)).float(), "error":torch.from_numpy(np.array(weight_list)).float()})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return np.asarray(data_list)


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=False):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        