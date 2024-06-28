from abc import ABC
import os
from typing import Optional, List, Union

import torch
from torch.utils.data import Dataset

from .intrinsics import Intrinsics

import numpy as np

class BaseDataset(Dataset, ABC):
    def __init__(self,
                 datadir: str,
                 scene_bbox: torch.Tensor,
                 split: str,
                 is_ndc: bool,
                 is_contracted: bool,
                 rays_o: Optional[torch.Tensor],
                 rays_d: Optional[torch.Tensor],
                 intrinsics: Union[Intrinsics, List[Intrinsics]],
                 batch_size: Optional[int] = None,
                 imgs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                 sampling_weights: Optional[torch.Tensor] = None,
                 weights_subsampled: int = 1,
                 num_imgs: Optional[int] = None,
                 is_robust_loss_enabled: Optional[bool] = False,
                 patch_size: Optional[int] = None,
                 ):
        self.datadir = datadir
        self.name = os.path.basename(self.datadir)
        self.scene_bbox = scene_bbox
        self.split = split
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.weights_subsampled = weights_subsampled
        self.batch_size = batch_size
        if self.split == 'train':
            assert self.batch_size is not None
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.imgs = imgs
        if self.imgs is not None:
            self.num_samples = len(self.imgs)
        elif self.rays_o is not None:
            self.num_samples = len(self.rays_o)
        else:
            self.num_samples = None
            #raise RuntimeError("Can't figure out num_samples.")
        self.intrinsics = intrinsics
        self.sampling_weights = sampling_weights
        if self.sampling_weights is not None:
            assert len(self.sampling_weights) == self.num_samples, (
                f"Expected {self.num_samples} sampling weights but given {len(self.sampling_weights)}."
            )
        self.sampling_batch_size = 2_000_000  # Increase this?
        if self.num_samples is not None:
            self.use_permutation = self.num_samples < 100_000_000  # 64M is static
        else:
            self.use_permutation = True
        self.perm = None
        self.num_imgs = num_imgs
        self.is_robust_loss_enabled = is_robust_loss_enabled
        self.patch_size = patch_size

    @property
    def img_h(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.height for i in self.intrinsics]
        return self.intrinsics.height

    @property
    def img_w(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.width for i in self.intrinsics]
        return self.intrinsics.width

    def reset_iter(self):
        if self.sampling_weights is None and self.use_permutation:
            self.perm = torch.randperm(self.num_samples)
        else:
            del self.perm
            self.perm = None

    def get_rand_ids(self, index):
        assert self.batch_size is not None, "Can't get rand_ids for test split"
        if self.sampling_weights is not None:
            batch_size = self.batch_size // (self.weights_subsampled ** 2)
            num_weights = len(self.sampling_weights)
            if num_weights > self.sampling_batch_size:
                # Take a uniform random sample first, then according to the weights
                subset = torch.randint(
                    0, num_weights, size=(self.sampling_batch_size,),
                    dtype=torch.int64, device=self.sampling_weights.device)
                samples = torch.multinomial(
                    input=self.sampling_weights[subset], num_samples=batch_size)
                return subset[samples]
            return torch.multinomial(
                input=self.sampling_weights, num_samples=batch_size)
        else:
            batch_size = self.batch_size
            if self.use_permutation:
                return self.perm[index * batch_size: (index + 1) * batch_size]
            else:
                return torch.randint(0, self.num_samples, size=(batch_size, ))

    def get_rand_patch(self):
        assert self.batch_size is not None, "Can't get rand_patches for test split"
        assert self.num_imgs is not None, "Selection of random patch needs number of total train images"
        batch_size = self.batch_size
        num_imgs = self.num_imgs
        img_h = self.img_h
        img_w = self.img_w

        assert torch.equal(torch.sqrt(torch.tensor(batch_size)), torch.sqrt(torch.tensor(batch_size)).round()), "Patch size must be quadratic"
        patch_height = int(torch.sqrt(torch.tensor(batch_size)).item())
        patch_width = int(torch.sqrt(torch.tensor(batch_size)).item())

        image_idx = torch.randint(0, num_imgs, size=(1,)).item()

        max_row_start = img_h - patch_height
        max_col_start = img_w - patch_width
        row_start = torch.randint(0, max_row_start + 1, (1,)).item()
        col_start = torch.randint(0, max_col_start + 1, (1,)).item()
        
        # Debug purposes
        # reshaped_imgs = self.imgs.clone().detach().view(num_imgs, img_h, img_w, 4)
        # patch = reshaped_imgs.numpy()[image_idx, row_start:row_start+patch_height, col_start:col_start+patch_width, :]
        # image = Image.fromarray((patch * 255).astype(np.uint8))
        # image.save('patch_image.png')

        index = []
        for r in range(patch_height):
            for c in range(patch_width):
                original_index = image_idx * (img_h * img_w) + (row_start + r) * img_w + (col_start + c)
                index.append(original_index)
        index = torch.tensor(index)

        return index
    
    def get_rand_patch_v2(self, image_idx):
        img_h = self.img_h
        img_w = self.img_w
        patch_size = self.patch_size

        max_row_start = img_h - patch_size
        max_col_start = img_w - patch_size
        row_start = torch.randint(0, max_row_start + 1, (1,)).item()
        col_start = torch.randint(0, max_col_start + 1, (1,)).item()

        index = []
        for r in range(patch_size):
            for c in range(patch_size):
                original_index = image_idx * (img_h * img_w) + (row_start + r) * img_w + (col_start + c)
                index.append(original_index)
        index = np.asarray(index)

        return index
    
    # TODO translate to numpy
    def rearrange_indices(self, blocks, num_patches_in_axis):
        end_block = []
        unit_block = []

        cnt = 0
        for block in blocks:
            if cnt < num_patches_in_axis:
                unit_block.append(block.copy())
                cnt += 1

                if cnt == num_patches_in_axis:
                    end_block.append(unit_block.copy())
                    unit_block = []
                    cnt = 0

        res = np.block(end_block)
        return res.flatten()

    
    def construct_random_patch(self, index):
        import math

        assert self.batch_size is not None, "Can't get rand_patches for test split"
        assert self.num_imgs is not None, "Selection of random patches requires number of total train images"
        assert self.patch_size is not None, "Selection of random patches requires size of a patch"

        batch_size = self.batch_size
        assert math.isqrt(batch_size), "Batch size must be quadratic"
        num_imgs = self.num_imgs
        patch_size = self.patch_size

        window_size = math.sqrt(batch_size)
        
        assert window_size % patch_size == 0, f"Patches of {patch_size}x{patch_size} don't fit perfectly into {int(window_size)}x{int(window_size)} window"

        num_patches_in_axis = int(window_size / patch_size)

        if patch_size == 1:
            return self.get_rand_ids(index)
        else:
            patches = []
            for i in range(num_patches_in_axis ** 2):
                image_idx = torch.randint(0, num_imgs, size=(1,)).item()
                patch_indices = self.get_rand_patch_v2(image_idx)
                patches.append(patch_indices.reshape(patch_size, patch_size))

            patches = np.asarray(patches)
            patches = self.rearrange_indices(patches, num_patches_in_axis)
            return patches

    def __len__(self):
        if self.split == 'train':
            return (self.num_samples + self.batch_size - 1) // self.batch_size
        else:
            return self.num_samples

    def __getitem__(self, index, return_idxs: bool = False):
        if self.split == 'train':
            if self.is_robust_loss_enabled:
                index = self.construct_random_patch(index)
            else:
                index = self.get_rand_ids(index)
        out = {}
        if self.rays_o is not None:
            out["rays_o"] = self.rays_o[index]
        if self.rays_d is not None:
            out["rays_d"] = self.rays_d[index]
        if self.imgs is not None: # [64000000, 4]
            out["imgs"] = self.imgs[index] #[4096, 4]
            # Debug purposes
            reshaped_imgs = out["imgs"].cpu().clone().detach().view(1, 64, 64, 4)
            patch = reshaped_imgs.numpy()[0, 0:64, 0:64, :]
            from PIL import Image
            image = Image.fromarray((patch * 255).astype(np.uint8))
            image.save('patch_image.png')
        else:
            out["imgs"] = None
        if return_idxs:
            return out, index
        return out
