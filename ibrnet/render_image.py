"""Functions for rendering a target view."""

from collections import OrderedDict
from ibrnet.render_ray import render_rays_mono
from ibrnet.render_ray import render_rays_mv
import torch


def render_single_image_nvi(
    frame_idx,
    time_embedding,
    time_offset,
    ray_sampler,
    ray_batch,
    model,
    projector,
    chunk_size,
    N_samples,
    args,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    render_stride=1,
    coarse_featmaps=None,
    fine_featmaps=None,
    is_train=True,
):
  """Render a target view for Nvidia dataset.

  Args:
    frame_idx: video frame index
    time_embedding: input time embedding
    time_offset: offset w.r.t reference time
    ray_sampler: target view ray sampler
    ray_batch: batch of ray information
    model: dynibar model
    projector: perspective projection module
    chunk_size: processing chunk size
    N_samples: number of coarse samples along the ray
    args: additional input arguments
    inv_uniform: use disparity-based sampling or not
    N_importance: number of fine samples along the ray
    det: deterministic sampling
    white_bkgd: whether background is present
    render_stride: pixel stride when rendering images
    coarse_featmaps: coarse-stage 2D feature map
    fine_featmaps: fine-stage 2D feature map
    is_train: is training or not

  Returns:
    outputs_fine_anchor: rendered fine images at target view from contents at
    nearby time
    outputs_fine_ref: rendered fine images at target view from contents at
    target time
    outputs_coarse_ref: rendered coarse images at target view from contents at
    target time
  """

  all_ret = OrderedDict([
      ('outputs_fine_anchor', OrderedDict()),
      ('outputs_fine_ref', OrderedDict()),
      ('outputs_coarse_ref', OrderedDict()),
  ])

  N_rays = ray_batch['ray_o'].shape[0]

  for i in range(0, N_rays, chunk_size):
    chunk = OrderedDict()
    for k in ray_batch:
      if ray_batch[k] is None:
        chunk[k] = None
      elif k in [
          'camera',
          'depth_range',
          'src_rgbs',
          'src_cameras',
          'anchor_src_rgbs',
          'anchor_src_cameras',
          'static_src_rgbs',
          'static_src_cameras',
      ]:
        chunk[k] = ray_batch[k]
      elif len(ray_batch[k].shape) == 3:  # flow and mask
        chunk[k] = ray_batch[k][:, i : i + chunk_size, ...]
      elif ray_batch[k] is not None:
        chunk[k] = ray_batch[k][i : i + chunk_size]
      else:
        chunk[k] = None

    ret = render_rays_mv(
        frame_idx=frame_idx,
        time_embedding=time_embedding,
        time_offset=time_offset,
        ray_batch=chunk,
        model=model,
        coarse_featmaps=coarse_featmaps,
        fine_featmaps=fine_featmaps,
        projector=projector,
        N_samples=N_samples,
        args=args,
        inv_uniform=inv_uniform,
        N_importance=N_importance,
        raw_noise_std=0.0,
        det=det,
        white_bkgd=white_bkgd,
        is_train=is_train,
    )

    # handle both coarse and fine outputs
    # cache chunk results on cpu
    if i == 0:
      for k in ret['outputs_coarse_ref']:
        all_ret['outputs_coarse_ref'][k] = []

      for k in ret['outputs_fine_ref']:
        all_ret['outputs_fine_ref'][k] = []

      if is_train:
        for k in ret['outputs_fine_anchor']:
          all_ret['outputs_fine_anchor'][k] = []

    for k in ret['outputs_coarse_ref']:
      all_ret['outputs_coarse_ref'][k].append(
          ret['outputs_coarse_ref'][k].cpu()
      )

    for k in ret['outputs_fine_ref']:
      all_ret['outputs_fine_ref'][k].append(ret['outputs_fine_ref'][k].cpu())

    if is_train:
      for k in ret['outputs_fine_anchor']:
        all_ret['outputs_fine_anchor'][k].append(
            ret['outputs_fine_anchor'][k].cpu()
        )

  rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[
      ::render_stride, ::render_stride, :
  ]
  # merge chunk results and reshape
  for k in all_ret['outputs_coarse_ref']:
    if k == 'random_sigma':
      continue

    if len(all_ret['outputs_coarse_ref'][k][0].shape) == 4:
      continue

    if len(all_ret['outputs_coarse_ref'][k][0].shape) == 3:
      tmp = torch.cat(all_ret['outputs_coarse_ref'][k], dim=1).reshape((
          all_ret['outputs_coarse_ref'][k][0].shape[0],
          rgb_strided.shape[0],
          rgb_strided.shape[1],
          -1,
      ))
    else:
      tmp = torch.cat(all_ret['outputs_coarse_ref'][k], dim=0).reshape(
          (rgb_strided.shape[0], rgb_strided.shape[1], -1)
      )
    all_ret['outputs_coarse_ref'][k] = tmp.squeeze()

  all_ret['outputs_coarse_ref']['rgb'][
      all_ret['outputs_coarse_ref']['mask'] == 0
  ] = 0.0

  # merge chunk results and reshape
  for k in all_ret['outputs_fine_ref']:
    if k == 'random_sigma':
      continue

    if len(all_ret['outputs_fine_ref'][k][0].shape) == 4:
      continue

    if len(all_ret['outputs_fine_ref'][k][0].shape) == 3:
      tmp = torch.cat(all_ret['outputs_fine_ref'][k], dim=1).reshape((
          all_ret['outputs_fine_ref'][k][0].shape[0],
          rgb_strided.shape[0],
          rgb_strided.shape[1],
          -1,
      ))
    else:
      tmp = torch.cat(all_ret['outputs_fine_ref'][k], dim=0).reshape(
          (rgb_strided.shape[0], rgb_strided.shape[1], -1)
      )
    all_ret['outputs_fine_ref'][k] = tmp.squeeze()

  all_ret['outputs_fine_ref']['rgb'][
      all_ret['outputs_fine_ref']['mask'] == 0
  ] = 0.0

  # merge chunk results and reshape
  if is_train:
    for k in all_ret['outputs_fine_anchor']:
      if k == 'random_sigma':
        continue

      if len(all_ret['outputs_fine_anchor'][k][0].shape) == 4:
        continue

      if len(all_ret['outputs_fine_anchor'][k][0].shape) == 3:
        tmp = torch.cat(all_ret['outputs_fine_anchor'][k], dim=1).reshape((
            all_ret['outputs_fine_anchor'][k][0].shape[0],
            rgb_strided.shape[0],
            rgb_strided.shape[1],
            -1,
        ))
      else:
        tmp = torch.cat(all_ret['outputs_fine_anchor'][k], dim=0).reshape(
            (rgb_strided.shape[0], rgb_strided.shape[1], -1)
        )
      all_ret['outputs_fine_anchor'][k] = tmp.squeeze()

    all_ret['outputs_fine_anchor']['rgb'][
        all_ret['outputs_fine_anchor']['mask'] == 0
    ] = 0.0

  all_ret['outputs_fine'] = None
  return all_ret


def render_single_image_mono(
    frame_idx,
    time_embedding,
    time_offset,
    ray_sampler,
    ray_batch,
    model,
    projector,
    chunk_size,
    N_samples,
    args,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    render_stride=1,
    featmaps=None,
    is_train=True,
    num_vv=2,
):
  """Render a target view for Monocular video.

  Args:
    frame_idx: video frame index
    time_embedding: input time embedding
    time_offset: offset w.r.t reference time
    ray_sampler: target view ray sampler
    ray_batch: batch of ray information
    model: dynibar model
    projector: perspective projection module
    chunk_size: processing chunk size
    N_samples: number of coarse samples along the ray
    args: additional input arguments
    inv_uniform: use disparity-based sampling or not
    N_importance: number of fine samples along the ray
    det: deterministic sampling
    white_bkgd: whether background is present
    render_stride: pixel stride when rendering images
    featmaps: coarse-stage 2D feature map
    is_train: is training or not
    num_vv: number of virtual source views used

  Returns:
    outputs_coarse_ref: rendered images at target view from combined contents at
    target time, coarse model
    outputs_coarse_st: rendered images at target view from static 
    contents at target time, coarse model
    outputs_coarse_anchor: cross-rendered images at target view 
    from combined contents at nearby time, coarse model

  """

  all_ret = OrderedDict([
      ('outputs_coarse_ref', OrderedDict()),
      ('outputs_coarse_st', OrderedDict()),
      ('outputs_coarse_anchor', OrderedDict()),
  ])

  N_rays = ray_batch['ray_o'].shape[0]

  for i in range(0, N_rays, chunk_size):
    chunk = OrderedDict()
    for k in ray_batch:
      if ray_batch[k] is None:
        chunk[k] = None
      elif k in [
          'camera',
          'anchor_camera',
          'depth_range',
          'src_rgbs',
          'src_cameras',
          'anchor_src_rgbs',
          'anchor_src_cameras',
          'static_src_rgbs',
          'static_src_cameras',
      ]:
        chunk[k] = ray_batch[k]
      elif len(ray_batch[k].shape) == 3:  # flow and mask
        chunk[k] = ray_batch[k][:, i : i + chunk_size, ...]
      elif ray_batch[k] is not None:
        chunk[k] = ray_batch[k][i : i + chunk_size]
      else:
        chunk[k] = None

    ret = render_rays_mono(
        frame_idx=frame_idx,
        time_embedding=time_embedding,
        time_offset=time_offset,
        ray_batch=chunk,
        model=model,
        featmaps=featmaps,
        projector=projector,
        N_samples=N_samples,
        args=args,
        inv_uniform=inv_uniform,
        N_importance=N_importance,
        raw_noise_std=0.0,
        det=det,
        white_bkgd=white_bkgd,
        is_train=is_train,
        num_vv=num_vv,
    )

    # handle both coarse and fine outputs
    # cache chunk results on cpu
    if i == 0:
      for k in ret['outputs_coarse_ref']:
        all_ret['outputs_coarse_ref'][k] = []

      for k in ret['outputs_coarse_st']:
        all_ret['outputs_coarse_st'][k] = []

      if is_train:
        for k in ret['outputs_coarse_anchor']:
          all_ret['outputs_coarse_anchor'][k] = []

      if ret['outputs_fine'] is None:
        all_ret['outputs_fine'] = None
      else:
        for k in ret['outputs_fine']:
          all_ret['outputs_fine'][k] = []

    for k in ret['outputs_coarse_ref']:
      all_ret['outputs_coarse_ref'][k].append(
          ret['outputs_coarse_ref'][k].cpu()
      )

    for k in ret['outputs_coarse_st']:
      all_ret['outputs_coarse_st'][k].append(ret['outputs_coarse_st'][k].cpu())

    if is_train:
      for k in ret['outputs_coarse_anchor']:
        all_ret['outputs_coarse_anchor'][k].append(
            ret['outputs_coarse_anchor'][k].cpu()
        )

    if ret['outputs_fine'] is not None:
      for k in ret['outputs_fine']:
        all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

  rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[
      ::render_stride, ::render_stride, :
  ]
  # merge chunk results and reshape
  for k in all_ret['outputs_coarse_ref']:
    if k == 'random_sigma':
      continue

    if len(all_ret['outputs_coarse_ref'][k][0].shape) == 4:
      continue

    if len(all_ret['outputs_coarse_ref'][k][0].shape) == 3:
      tmp = torch.cat(all_ret['outputs_coarse_ref'][k], dim=1).reshape((
          all_ret['outputs_coarse_ref'][k][0].shape[0],
          rgb_strided.shape[0],
          rgb_strided.shape[1],
          -1,
      ))
    else:
      tmp = torch.cat(all_ret['outputs_coarse_ref'][k], dim=0).reshape(
          (rgb_strided.shape[0], rgb_strided.shape[1], -1)
      )
    all_ret['outputs_coarse_ref'][k] = tmp.squeeze()

  all_ret['outputs_coarse_ref']['rgb'][
      all_ret['outputs_coarse_ref']['mask'] == 0
  ] = 0.0

  # merge chunk results and reshape
  for k in all_ret['outputs_coarse_st']:
    if k == 'random_sigma':
      continue

    if len(all_ret['outputs_coarse_st'][k][0].shape) == 4:
      continue

    if len(all_ret['outputs_coarse_st'][k][0].shape) == 3:
      tmp = torch.cat(all_ret['outputs_coarse_st'][k], dim=1).reshape((
          all_ret['outputs_coarse_st'][k][0].shape[0],
          rgb_strided.shape[0],
          rgb_strided.shape[1],
          -1,
      ))
    else:
      tmp = torch.cat(all_ret['outputs_coarse_st'][k], dim=0).reshape(
          (rgb_strided.shape[0], rgb_strided.shape[1], -1)
      )
    all_ret['outputs_coarse_st'][k] = tmp.squeeze()

  all_ret['outputs_coarse_st']['rgb'][
      all_ret['outputs_coarse_st']['mask'] == 0
  ] = 0.0

  # merge chunk results and reshape
  if is_train:
    for k in all_ret['outputs_coarse_anchor']:
      if k == 'random_sigma':
        continue

      if len(all_ret['outputs_coarse_anchor'][k][0].shape) == 4:
        continue

      if len(all_ret['outputs_coarse_anchor'][k][0].shape) == 3:
        tmp = torch.cat(all_ret['outputs_coarse_anchor'][k], dim=1).reshape((
            all_ret['outputs_coarse_anchor'][k][0].shape[0],
            rgb_strided.shape[0],
            rgb_strided.shape[1],
            -1,
        ))
      else:
        tmp = torch.cat(all_ret['outputs_coarse_anchor'][k], dim=0).reshape(
            (rgb_strided.shape[0], rgb_strided.shape[1], -1)
        )
      all_ret['outputs_coarse_anchor'][k] = tmp.squeeze()

    all_ret['outputs_coarse_anchor']['rgb'][
        all_ret['outputs_coarse_anchor']['mask'] == 0
    ] = 0.0

  return all_ret
