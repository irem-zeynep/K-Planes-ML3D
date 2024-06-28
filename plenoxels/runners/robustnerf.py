"""Computes RobustNeRF mask."""
from typing import Mapping, Tuple

from jax import lax
import numpy as np
import jax.numpy as jnp
import torch


def robustnerf_mask(errors, loss_threshold: float, config) -> Tuple[np.ndarray, Mapping[str, np.ndarray]]:
  """Computes RobustNeRF mask.

  Args:
    errors: f32[n,h,w,c]. Per-subpixel errors in a batch of patches.
    loss_threshold: f32[]. Upper bound on per-pixel loss to use to determine
      if a pixel is an inlier or not.
    config: Config object. A dictionary of hyperparameters.

  Returns:
    mask: f32[n,h,w,c or 1]. Binary mask that broadcasts to shape [n,h,w,c].
    stats: { str: f32[] }. Statistics to pass on.
  """
  epsilon = 1e-3
  #errors = jnp.array(errors.cpu().detach().numpy()) # [4096, 3]
  batch_size, _ = errors.shape
  errors = jnp.array(errors.view(1, config['patch_size2'], config['patch_size2'], 3).cpu().detach().numpy())

  error_dtype = errors.dtype

  ### Takes the average of the RGB values. f32[n,h,w,c] -> f32[n,h,w,1]
  error_per_pixel = jnp.mean(errors, axis=-1, keepdims=True)  # f32[n,h,w,1]

  ### Computes the value in 'error_per_pixel' corresponding to the quantile boundary set in config
  next_loss_threshold = jnp.quantile(error_per_pixel, config['inlier_quantile'])
  mask = jnp.ones_like(error_per_pixel, dtype=error_dtype)
  stats = {
      'loss_threshold': next_loss_threshold,
  }

  # Inlier pixels have a value of 1.0 in the mask.
  is_inlier_pixel = (error_per_pixel < loss_threshold).astype(error_dtype)
  ### Computes the fraction of inlier pixels
  stats['is_inlier_loss'] = jnp.mean(is_inlier_pixel, dtype=error_dtype)


  # Apply fxf (3x3) box filter 'window' for smoothing (diffusion).
  f = config['smoothed_filter_size']

  ### Creates a box filter
  window = jnp.ones((1, 1, f, f)) / (f * f)

  ### The transpose aligns 'is_inlier_pixel' to the 'window' shape, by changing the axes according to [0, 3, 1, 2]
  ### 0 and 3 axes are the batch and color dimensions
  has_inlier_neighbors = lax.conv(
        jnp.transpose(is_inlier_pixel, [0, 3, 1, 2]), window, (1, 1), 'SAME'
    )
  
  ### The transpose changes back the dimensions as it was before the diffusion step
  has_inlier_neighbors = jnp.transpose(has_inlier_neighbors, [0, 2, 3, 1])

  # Binarize after smoothing.
  # config.robustnerf_smoothed_inlier_quantile default is 0.5 which means at
  # least 50% of neighbouring pixels are inliers.
  has_inlier_neighbors = (
      has_inlier_neighbors > 1 - config['smoothed_inlier_quantile']
  ).astype(error_dtype)
  stats['has_inlier_neighbors'] = jnp.mean(has_inlier_neighbors)

  is_inlier_pixel = (
      has_inlier_neighbors + is_inlier_pixel > epsilon
  ).astype(error_dtype)
 
  # Construct binary mask for inner pixels. The entire inner patch is either
  # active or inactive.
  # patch_size is the input patch (h,w), inner patch size can be any value
  # smaller than patch_size. Default is for the inner patch size to be half
  # the input patch size (i.e. 16x16 -> 8x8).
  inner_patch_mask = _robustnerf_inner_patch_mask(
      config['inner_patch_size'], config['patch_size2']
  )

  is_inlier_patch = jnp.mean(
      is_inlier_pixel, axis=[0], keepdims=True
  )  # f32[n,1,1,1]
  # robustnerf_inner_patch_inlier_quantile what percentage of the patch
  # should be inliers so that the patch is counted as an inlier patch.
  is_inlier_patch = (
      is_inlier_patch > 1 - config['inner_patch_inlier_quantile']
  ).astype(error_dtype)

  is_inlier_patch = is_inlier_patch * inner_patch_mask
  stats['is_inlier_patch'] = jnp.mean(is_inlier_patch)

  # A pixel is an inlier if it is an inlier according to any of the above
  # criteria.
  mask = (
      is_inlier_patch + is_inlier_pixel > epsilon
  ).astype(error_dtype)

  stats['mask'] = jnp.mean(mask)

  # TODO replace hard coded values
  mask = np.array(mask).reshape((batch_size, 1))
  mask = torch.from_numpy(np.array(mask))
  return mask, stats


def _robustnerf_inner_patch_mask(
    inner_patch_size, outer_patch_size, *, dtype=np.float32
):
  """Constructs binary mask for inner patch.

  Args:
    inner_patch_size: Size of the (square) inside patch.
    outer_patch_size: Size of the (square) outer patch.
    dtype: dtype for result

  Returns:
    Binary mask of shape (1, outer_patch_size, outer_patch_size, 1). Mask is
      1.0 for the center (inner_patch_size, inner_patch_size) square and 0.0
      elsewhere.
  """
  pad_size_lower = (outer_patch_size - inner_patch_size) // 2
  pad_size_upper = outer_patch_size - (inner_patch_size + pad_size_lower)
  mask = jnp.pad(
      jnp.ones((1, inner_patch_size, inner_patch_size, 1), dtype=dtype),
      (
          (0, 0),  # batch
          (pad_size_lower, pad_size_upper),  # height
          (pad_size_lower, pad_size_upper),  # width
          (0, 0),  # channels
      ),
  )
  return mask

