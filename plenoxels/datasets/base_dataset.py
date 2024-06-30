from abc import ABC
import os
from typing import Optional, List, Union

import torch
from torch.utils.data import Dataset

from .intrinsics import Intrinsics

import numpy as np
import math
import time

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

        if type(self.img_h) is list:
            self.img_start_index = []
            for j in range(len(self.img_h)):
                current_img_start = 0
                for i in range(j):
                    current_img_start += self.img_h[i] * self.img_w[i]
                self.img_start_index.append(current_img_start)


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

    # Get random patches from images with same sizes
    def get_rand_patch_static(self, image_idx):
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
    
    # Get random patches from images with dynamic sizes
    def get_rand_patch_dynamic(self, image_idx):
        img_h = self.img_h[image_idx]
        img_w = self.img_w[image_idx]
        patch_size = self.patch_size

        max_row_start = img_h - patch_size
        max_col_start = img_w - patch_size
        row_start = torch.randint(0, max_row_start + 1, (1,)).item()
        col_start = torch.randint(0, max_col_start + 1, (1,)).item()

        index = []
        for r in range(patch_size):
            for c in range(patch_size):
                original_index = self.img_start_index[image_idx] + (row_start + r) * img_w + (col_start + c)
                index.append(original_index)
        index = np.asarray(index)

        return index
    
    def construct_random_patch(self, index):
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
                # PhotoTourism e.g. has images of different sizes.
                if type(self.img_h) is list:
                    patch_indices = self.get_rand_patch_dynamic(image_idx)
                else:
                    patch_indices = self.get_rand_patch_static(image_idx)
                patches.append(patch_indices)

            patches = np.asarray(patches)
            patches = patches.flatten()
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

            # # Debug purposes
            # from PIL import Image
            # reshaped_patches = out["imgs"].cpu().clone().detach().view(4, 64, 64, 3).numpy()
            # for i, patch in enumerate(reshaped_patches):
            #     patch = Image.fromarray((patch).astype(np.uint8))
            #     patch.save(f'test/output/patch_image_{i}.png')
        else:
            out["imgs"] = None
        if return_idxs:
            return out, index
        return out
