"""Main helper functions for rendering rays.

Modified from
https://github.com/googleinterns/IBRNet/blob/master/ibrnet/render_ray.py.
"""

import numpy as np

from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

USE_DISTANCE = False
USE_SOFTPLUS = True
VARIANCE_CLAMP = False


def sample_pdf(bins, weights, N_samples, det=False):
  """Sample PDF from coarse samples. Same as original NeRF."""

  M = weights.shape[1]
  weights += 1e-5
  # Get pdf
  pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
  cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
  cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)  # [N_rays, M+1]

  # Take uniform samples
  if det:
    u = torch.linspace(0.0, 1.0, N_samples, device=bins.device)
    u = u.unsqueeze(0).repeat(bins.shape[0], 1)  # [N_rays, N_samples]
  else:
    u = torch.rand(bins.shape[0], N_samples, device=bins.device)

  # Invert CDF
  above_inds = torch.zeros_like(u, dtype=torch.long)  # [N_rays, N_samples]
  for i in range(M):
    above_inds += (u >= cdf[:, i : i + 1]).long()

  # random sample inside each bin
  below_inds = torch.clamp(above_inds - 1, min=0)
  inds_g = torch.stack(
      (below_inds, above_inds), dim=2
  )  # [N_rays, N_samples, 2]

  cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
  cdf_g = torch.gather(
      input=cdf, dim=-1, index=inds_g
  )  # [N_rays, N_samples, 2]

  bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
  bins_g = torch.gather(
      input=bins, dim=-1, index=inds_g
  )  # [N_rays, N_samples, 2]

  # fix numeric issue
  denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_rays, N_samples]
  denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
  t = (u - cdf_g[:, :, 0]) / denom

  samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

  return samples


def sample_along_camera_ray(
    ray_o, ray_d, depth_range, N_samples, inv_uniform=False, det=False
):
  """Create samples along a ray.

  Args:
    ray_o: ray origin
    ray_d: ray direction
    depth_range: depth bounds
    N_samples: number of samples
    inv_uniform: whether the sample according to inverse depth
    det: deterministic sample or not

  Returns:
    pts: 3D sample point location
    z_vals: depth value of samples
    s_vals: normalized depth value of samples
  """

  # will sample inside [near_depth, far_depth]
  # assume the nearest possible depth is at least (min_ratio * depth)
  near_depth_value = depth_range[0, 0]
  far_depth_value = depth_range[0, 1]
  assert (
      near_depth_value > 0
      and far_depth_value > 0
      and far_depth_value > near_depth_value
  )

  near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

  far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
  if inv_uniform:
    start = 1.0 / near_depth  # [N_rays,]
    step = (1.0 / far_depth - start) / (N_samples - 1)
    inv_z_vals = torch.stack(
        [start + i * step for i in range(N_samples)], dim=1
    )  # [N_rays, N_samples]
    z_vals = 1.0 / inv_z_vals
  else:
    start = near_depth
    step = (far_depth - near_depth) / (N_samples - 1)
    z_vals = torch.stack(
        [start + i * step for i in range(N_samples)], dim=1
    )  # [N_rays, N_samples]

  if not det:
    # get intervals between samples
    mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
    upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
    lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
    # uniformly random samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

  ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
  ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
  pts = z_vals.unsqueeze(2) * ray_d + ray_o  # [N_rays, N_samples, 3]

  # from mip-nerf 360 normalized distance
  s_vals = ((1.0 / z_vals) - (1.0 / near_depth_value)) / (
      1.0 / far_depth_value - 1.0 / near_depth_value
  )

  return pts, z_vals, s_vals


def raw2outputs_vanilla(raw, z_vals, mask):
  """Convert raw output to color and density without composition.

  Args:
    raw: raw data
    z_vals: depth value of samples
    mask: valid sample

  Returns:
    rgb: rgb color of the ray
    depth: depth of the ray
    mask: valid mask of the ray
    alpha: alpha value of the ray
    z_vals: sample depth
  """

  rgb = raw[:, :, :3]  # [N_rays, N_samples, 3]
  sigma = raw[:, :, 3]  # [N_rays, N_samples]

  # USE SOFTPLUS for more stable density prediction.
  if USE_SOFTPLUS:
    sigma2alpha = (
        lambda sigma, dists, act_fn=torch.nn.Softplus(): 1.0
        - torch.exp(-act_fn(sigma) * dists)
    )
  else:
    sigma2alpha = lambda sigma, dists, act_fn=F.relu: 1.0 - torch.exp(
        -act_fn(sigma) * dists
    )

  # Following IBRNet, we don't use the interval distance.
  if USE_DISTANCE:
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [
            dists,
            torch.Tensor([1e10]).expand(dists[..., :1].shape).to(raw.device),
        ],
        -1,
    )  # [N_rays, N_samples]
  else:
    dists = torch.ones_like(z_vals[..., 1:])
    dists = torch.cat(
        [
            dists,
            torch.Tensor([1e10]).expand(dists[..., :1].shape).to(raw.device),
        ],
        -1,
    )  # [N_rays, N_samples]

  alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

  # Eq. (3): T
  T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)[:, :-1]
  T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

  # maths show weights and summation of weights are always inside [0, 1]
  weights = alpha * T  # [N_rays, N_samples]

  rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

  # should at least have 8 valid observation on the ray,
  # otherwise don't consider its loss
  mask = (
      mask.float().sum(dim=1) > 8
  )
  depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays,]

  ret = OrderedDict([
      ('rgb', rgb_map),
      ('depth', depth_map),
      ('weights', weights),  # used for importance sampling of fine samples
      ('mask', mask),
      ('alpha', alpha),
      ('z_vals', z_vals),
  ])

  return ret


def raw2outputs(
    raw_dy,
    raw_static,
    z_vals,
    mask_dy,
    mask_static,
    raw_noise_std=0.0,
):
  """Convert raw output to color and density with composition.

  Args:
    raw_dy: raw data from dynamic model
    raw_static: ray data from static model
    z_vals: sample depth
    mask_dy: valid mask for dynamic model
    mask_static: valid mask for static model
    raw_noise_std: noise std added to the density

  Returns:
      rgb: composited rgb color of ray
      rgb: rgb color of static component
      rgb_dy: rgb color of dynamic component
      alpha_dy: alpha of dynamic component
      weights_dy: weight of dynamic component
      weights_st: weight of static component
      alpha: alpha of composited model
      weights: weights of composited model
      mask: mask of composited model
      z_vals: depth value of composited model

  """

  rgb_dy = raw_dy[:, :, :3]  # [N_rays, N_samples, 3]
  sigma_dy = raw_dy[:, :, 3]  # [N_rays, N_samples]

  rgb_static = raw_static[:, :, :3]  # [N_rays, N_samples, 3]
  sigma_static = raw_static[:, :, 3]  # [N_rays, N_samples]

  # Use softplus for stable training of density prediction.
  if USE_SOFTPLUS:
    sigma2alpha = (
        lambda sigma, dists, act_fn=torch.nn.Softplus(): 1.0
        - torch.exp(-act_fn(sigma) * dists)
    )
  else:
    sigma2alpha = lambda sigma, dists, act_fn=F.relu: 1.0 - torch.exp(
        -act_fn(sigma) * dists
    )

  # Following IBRNet, we don't interval distance for rendering.
  if USE_DISTANCE:
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [
            dists,
            torch.Tensor([1e10]).expand(dists[..., :1].shape).to(raw_dy.device),
        ],
        -1,
    )  # [N_rays, N_samples]
  else:
    dists = torch.ones_like(z_vals[..., 1:])
    dists = torch.cat(
        [
            dists,
            torch.Tensor([1e10]).expand(dists[..., :1].shape).to(raw_dy.device),
        ],
        -1,
    )  # [N_rays, N_samples]

  alpha_dy = sigma2alpha(sigma_dy, dists)  # [N_rays, N_samples]
  alpha_static = sigma2alpha(sigma_static, dists)  # [N_rays, N_samples]

  # Eq. (3): T
  alpha = 1 - (1 - alpha_static) * (1 - alpha_dy)

  T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)[:, :-1]
  T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

  # summation of weights along a ray are always inside [0, 1]
  weights_dy = alpha_dy * T  # [N_rays, N_samples]

  depth_map_dy = torch.sum(
      weights_dy * z_vals, dim=-1, keepdims=True
  )  # [N_rays,]

  rgb_map_dy = torch.sum(weights_dy.unsqueeze(2) * rgb_dy, dim=1)  # [N_rays, 3]

  weights_static = alpha_static * T  # [N_rays, N_samples]
  rgb_map_static = torch.sum(
      weights_static.unsqueeze(2) * rgb_static, dim=1
  )  # [N_rays, 3]

  rgb_map = rgb_map_dy + rgb_map_static

  weights = alpha * T  # (N_rays, N_samples_)
  # should at least have 8 valid observation on the ray,
  # otherwise don't consider its loss
  mask = torch.bitwise_or(
      mask_dy.float().sum(dim=1) > 8, mask_static.float().sum(dim=1) > 8
  )
  depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays,]

  ret = OrderedDict([
      ('rgb', rgb_map),
      ('rgb_static', rgb_map_static),
      ('rgb_dy', rgb_map_dy),
      ('depth', depth_map),
      ('alpha_dy', alpha_dy),
      ('weights_dy', weights_dy),
      ('weights_st', weights_static),
      ('alpha', alpha),
      ('weights', weights),  # used for importance sampling of fine samples
      ('mask', mask),
      ('z_vals', z_vals),
  ])

  return ret


def compute_optical_flow(outputs_coarse, raw_pts_3d_seq, src_cameras, uv_grid):
  """Derive 2D optical from 3D scene flow."""
  src_cameras = src_cameras.squeeze(0)  # [n_views, 34]

  src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
  src_c2w = src_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
  src_w2c = torch.inverse(src_c2w)

  weights = outputs_coarse['weights'][None, ..., None]

  exp_pts_3d_seq = torch.sum(weights * raw_pts_3d_seq, dim=-2).unsqueeze(-1)

  exp_pts_3d_seq_src = (
      torch.matmul(src_w2c[:, None, :3, :3], (exp_pts_3d_seq))
      + src_w2c[:, None, :3, 3:4]
  )
  exp_pix_time_seq_src = torch.matmul(
      src_intrinsics[:, None, :3, :3], exp_pts_3d_seq_src
  )
  exp_pix_time_seq_src = (
      exp_pix_time_seq_src / exp_pix_time_seq_src[:, :, -1:, :]
  )

  render_flow = exp_pix_time_seq_src[..., :2, 0] - uv_grid[None, ...]

  return render_flow


def compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, trajectory_basis_i):
  return torch.cat(
      [
          torch.sum(raw_coeff_x * trajectory_basis_i, axis=-1, keepdim=True),
          torch.sum(raw_coeff_y * trajectory_basis_i, axis=-1, keepdim=True),
          torch.sum(raw_coeff_z * trajectory_basis_i, axis=-1, keepdim=True),
      ],
      dim=-1,
  )


def compute_ref_plucker_coordinate(ray_o, ray_d):
  """Compute plucker coordinate for rays from target views."""
  input_ray_d = F.normalize(ray_d, dim=-1)
  moment = torch.cross(ray_o, input_ray_d)

  return torch.cat([input_ray_d, moment], axis=-1)


def compute_src_plucker_coordinate(pts, src_cameras):
  """Compute plucker coordinate for rays from source views."""
  src_poses = src_cameras[0, :, -16:].reshape(-1, 4, 4)
  ray_o = src_poses[:, :3, 3].unsqueeze(1).unsqueeze(1)

  if len(pts.shape) == 3:
    ray_src = pts.unsqueeze(0) - ray_o
  else:
    ray_src = pts - ray_o

  ray_src = F.normalize(ray_src, dim=-1)

  moment_src = torch.cross(
      ray_o.expand(-1, ray_src.shape[1], ray_src.shape[2], -1), ray_src
  )

  return torch.cat([ray_src, moment_src], axis=-1).permute(1, 2, 0, 3)


def z_to_s(z_vals, near_depth_value, far_depth_value):
  """Normalize ray depth based on Mip-NeRF-360."""
  s_vals = ((1.0 / z_vals) - (1.0 / near_depth_value)) / (
      1.0 / far_depth_value - 1.0 / near_depth_value
  )
  return s_vals


def fine_render_rays(
    projector,
    ray_batch,
    featmaps,
    pts_ref,
    z_vals,
    s_vals,
    ref_time_embedding,
    anchor_time_embedding,
    ref_frame_idx,
    anchor_frame_idx,
    ref_time_offset,
    anchor_time_offset,
    net_dy,
    net_st,
    motion_mlp,
    trajectory_basis,
    occ_weights_mode,
    is_train,
):
  """Rendering rays for fine model. Used for dynamic scene datasets.

  Args:
    projector: perspective projection module
    ray_batch: input ray information
    featmaps: 2D source view feature maps
    pts_ref: reference time 3D point samples
    z_vals: sample depths
    s_vals: sample normalized depths
    ref_time_embedding: reference time embedding
    anchor_time_embedding: nearby time embeeding
    ref_frame_idx: reference video frame index
    anchor_frame_idx: nearby video frame index
    ref_time_offset: offset w.r.t reference time
    anchor_time_offset: offset w.r.t nearby time
    net_dy: dynamic model
    net_st: static model
    motion_mlp: motion trajectory MLP
    trajectory_basis: trajectory basis
    occ_weights_mode: occulution weight mode
    is_train: is training or node

  Returns:
    outputs_ref: composited model outputs at reference time
    outputs_ref_dy: dynamic model outputs at reference time
    outputs_anchor: composited model outputs from nearby time
    outputs_anchor_dy: dynamic model outputs from nearby time
  """
  input_ray_dir = F.normalize(ray_batch['ray_d'], dim=-1)
  num_frames = int(ref_frame_idx / ref_time_embedding)

  N_rays, N_samples = pts_ref.shape[:2]
  num_last_samples = int(round(N_samples * 0.1))

  # Scene motion at reference time in global coordidate system.
  ref_time_embedding_ = ref_time_embedding[None, None, :].repeat(
      N_rays, N_samples, 1
  )
  # 3D point locations in global coordindate system and reference time
  ref_xyzt = (
      torch.cat([pts_ref, ref_time_embedding_], dim=-1)
      .float()
      .to(pts_ref.device)
  )
  raw_coeff_xyz = motion_mlp(ref_xyzt)
  raw_coeff_xyz[:, -num_last_samples:, :] *= 0.0

  num_basis = trajectory_basis.shape[1]
  raw_coeff_x = raw_coeff_xyz[..., 0:num_basis]
  raw_coeff_y = raw_coeff_xyz[..., num_basis : num_basis * 2]
  raw_coeff_z = raw_coeff_xyz[..., num_basis * 2 : num_basis * 3]

  ref_traj_pts_dict = {}
  # We always aggregate 6 nearby source views for dynamic model.
  for offset in [-3, -2, -1, 0, 1, 2, 3]:
    traj_pts_ref = compute_traj_pts(
        raw_coeff_x,
        raw_coeff_y,
        raw_coeff_z,
        trajectory_basis[None, None, ref_frame_idx + offset, :],
    )

    ref_traj_pts_dict[offset] = traj_pts_ref

  pts_3d_seq_ref = []
  for offset in ref_time_offset:
    pts_3d_seq_ref.append(
        pts_ref + (ref_traj_pts_dict[offset] - ref_traj_pts_dict[0])
    )

  pts_3d_seq_ref = torch.stack(pts_3d_seq_ref, 0)
  pts_3d_static = pts_ref[None, ...].repeat(
      ray_batch['static_src_rgbs'].shape[1], 1, 1, 1
  )

  # Feature query from source view with scene motion, rendering to target view.
  rgb_feat_ref, ray_diff_ref, mask_ref = projector.compute_with_motions(
      pts_ref,
      pts_3d_seq_ref,
      ray_batch['camera'],
      ray_batch['src_rgbs'],
      ray_batch['src_cameras'],
      featmaps=featmaps[0],
  )  # [N_rays, N_samples, N_views, x]
  # Feature query from source view without scene motion, used for static model.
  rgb_feat_static, ray_diff_static, mask_static = (
      projector.compute_with_motions(
          pts_ref,
          pts_3d_static,
          ray_batch['camera'],
          ray_batch['static_src_rgbs'],
          ray_batch['static_src_cameras'],
          featmaps=featmaps[2],
      )
  )  # [N_rays, N_samples, N_views, x]

  # valid masks 
  pixel_mask_ref = (
      mask_ref[..., 0].sum(dim=2) > 1
  )  # [N_rays, N_samples], should at least have 2 observations
  pixel_mask_static = (
      mask_static[..., 0].sum(dim=2) > 1
  )  # [N_rays, N_samples], should at least have 2 observations

  ref_time_diff = torch.from_numpy(np.array(ref_time_offset)).to(
      rgb_feat_ref.device
  ) / float(num_frames)
  ref_time_diff = ref_time_diff[None, None, :, None].expand(
      ray_diff_ref.shape[0], ray_diff_ref.shape[1], -1, -1
  )
  # Obtain per-sample color and density
  raw_coarse_ref = net_dy(
      pts_ref,
      rgb_feat_ref,
      input_ray_dir,
      ray_diff_ref,
      ref_time_diff,
      mask_ref,
      ref_time_embedding_,
  )  # [N_rays, N_samples, 4]
  # Compute ray coordinates from reference view
  ref_rays_coords = compute_ref_plucker_coordinate(
      ray_batch['ray_o'], ray_batch['ray_d']
  )
  # Compute ray coordinates from source view
  src_rays_coords = compute_src_plucker_coordinate(
      pts_ref, ray_batch['static_src_cameras']
  )

  raw_static = net_st(
      pts_ref,
      ref_rays_coords,
      src_rays_coords,
      rgb_feat_static,
      input_ray_dir,
      ray_diff_static,
      mask_static,
  )
  # Obtain per-sample color and density at reference view composited from
  # static and dynamic models.
  outputs_ref = raw2outputs(
      raw_coarse_ref, raw_static, z_vals, pixel_mask_ref, pixel_mask_static
  )
  # Obtain per-sample color and density at reference view,
  # from dynamic model only.
  outputs_ref_dy = raw2outputs_vanilla(raw_coarse_ref, z_vals, pixel_mask_ref)

  render_flows = compute_optical_flow(
      outputs_ref,
      pts_3d_seq_ref,
      ray_batch['src_cameras'],
      ray_batch['uv_grid'],
  )
  outputs_ref['render_flows'] = render_flows
  outputs_ref['s_vals'] = s_vals

  outputs_anchor = None
  outputs_anchor_dy = None
  exp_sf_p = torch.sum(
      outputs_ref['weights'][..., None]
      * (ref_traj_pts_dict[2] - ref_traj_pts_dict[0]),
      dim=-2,
  )
  exp_sf_m = torch.sum(
      outputs_ref['weights'][..., None]
      * (ref_traj_pts_dict[-2] - ref_traj_pts_dict[0]),
      dim=-2,
  )
  outputs_ref['exp_sf'] = torch.max(exp_sf_p, exp_sf_m)

  return outputs_ref, outputs_ref_dy, outputs_anchor, outputs_anchor_dy


def render_rays_mv(
    frame_idx,
    time_embedding,
    time_offset,
    ray_batch,
    model,
    projector,
    coarse_featmaps,
    fine_featmaps,
    N_samples,
    args,
    inv_uniform=False,
    N_importance=0,
    raw_noise_std=0.0,
    det=False,
    white_bkgd=False,
    is_train=True,
):
  """Render both coarse and fine samples, used for dynamic scene dataset.

  Args:
    frame_idx: reference video frame index
    time_embedding: reference time embedding
    time_offset: offset w.r.t reference time
    ray_batch: input ray information
    model: dynibar model
    projector: perspective projection module
    coarse_featmaps: coarse model 2D feature maps
    fine_featmaps: fine model 2D feature maps
    N_samples: number of coarse samples
    args: input argument list
    inv_uniform: dispairty sampling
    N_importance: number of fine samples
    raw_noise_std: noise standard devition added to density output
    det: deterministic sampling
    white_bkgd: having white background?
    is_train: is training or not?

  Returns:
    outputs_coarse_ref: coarse model composited output from reference time
    outputs_fine_ref: fine model composited output from reference time
    outputs_fine_ref_dy: fine dynamic model output from reference time
    outputs_fine_anchor: fine composited output from nearby time
    outputs_fine_anchor_dy: fine dynamic model output from nearby time
  """

  ref_frame_idx, anchor_frame_idx = frame_idx[0], frame_idx[1]
  ref_time_embedding, anchor_time_embedding = (
      time_embedding[0],
      time_embedding[1],
  )
  ref_time_offset, anchor_time_offset = time_offset[0], time_offset[1]
  num_frames = int(ref_frame_idx / ref_time_embedding)

  input_ray_dir = F.normalize(ray_batch['ray_d'], dim=-1)

  ret = {'outputs_coarse': None, 'outputs_fine': None}

  # pts: [N_rays, N_samples, 3]
  # z_vals: [N_rays, N_samples]
  pts_ref, z_vals, _ = sample_along_camera_ray(
      ray_o=ray_batch['ray_o'],
      ray_d=ray_batch['ray_d'],
      depth_range=ray_batch['depth_range'],
      N_samples=N_samples,
      inv_uniform=inv_uniform,
      det=det,
  )

  N_rays, N_samples = pts_ref.shape[:2]
  num_last_samples = int(round(N_samples * 0.1))

  with torch.no_grad():
    # Scene motion at reference time in global coordindate system.
    ref_time_embedding_ = ref_time_embedding[None, None, :].repeat(
        N_rays, N_samples, 1
    )

    ref_xyzt = (
        torch.cat([pts_ref, ref_time_embedding_], dim=-1)
        .float()
        .to(pts_ref.device)
    )
    raw_coeff_xyz = model.motion_mlp(ref_xyzt)
    raw_coeff_xyz[:, -num_last_samples:, :] *= 0.0

    num_basis = model.trajectory_basis.shape[1]
    raw_coeff_x = raw_coeff_xyz[..., 0:num_basis]
    raw_coeff_y = raw_coeff_xyz[..., num_basis : num_basis * 2]
    raw_coeff_z = raw_coeff_xyz[..., num_basis * 2 : num_basis * 3]

    ref_traj_pts_dict = {}
    # Always use 6 nearby source views for dynamic model.
    for offset in [-3, -2, -1, 0, 1, 2, 3]:
      traj_pts_ref = compute_traj_pts(
          raw_coeff_x,
          raw_coeff_y,
          raw_coeff_z,
          model.trajectory_basis[None, None, ref_frame_idx + offset, :],
      )

      ref_traj_pts_dict[offset] = traj_pts_ref

    pts_3d_seq_ref = []
    for offset in ref_time_offset:
      pts_3d_seq_ref.append(
          pts_ref + (ref_traj_pts_dict[offset] - ref_traj_pts_dict[0])
      )

    pts_3d_seq_ref = torch.stack(pts_3d_seq_ref, 0)
    pts_3d_static = pts_ref[None, ...].repeat(
        ray_batch['static_src_rgbs'].shape[1], 1, 1, 1
    )

    # feature query from source view with scene motions, for ref view
    rgb_feat_ref, ray_diff_ref, mask_ref = projector.compute_with_motions(
        pts_ref,
        pts_3d_seq_ref,
        ray_batch['camera'],
        ray_batch['src_rgbs'],
        ray_batch['src_cameras'],
        featmaps=coarse_featmaps[0],
    )  # [N_rays, N_samples, N_views, x]

    rgb_feat_static, ray_diff_static, mask_static = (
        projector.compute_with_motions(
            pts_ref,
            pts_3d_static,
            ray_batch['camera'],
            ray_batch['static_src_rgbs'],
            ray_batch['static_src_cameras'],
            featmaps=coarse_featmaps[2],
        )
    )  # [N_rays, N_samples, N_views, x]

    pixel_mask_ref = (
        mask_ref[..., 0].sum(dim=2) > 1
    )  # [N_rays, N_samples], should at least have 2 observations
    pixel_mask_static = (
        mask_static[..., 0].sum(dim=2) > 1
    )  # [N_rays, N_samples], should at least have 2 observations

    ref_time_diff = torch.from_numpy(np.array(ref_time_offset)).to(
        rgb_feat_ref.device
    ) / float(num_frames)
    ref_time_diff = ref_time_diff[None, None, :, None].expand(
        ray_diff_ref.shape[0], ray_diff_ref.shape[1], -1, -1
    )

    raw_coarse_ref = model.net_coarse_dy(
        pts_ref,
        rgb_feat_ref,
        input_ray_dir,
        ray_diff_ref,
        ref_time_diff,
        mask_ref,
        ref_time_embedding_,
    )  # [N_rays, N_samples, 4]

    ref_rays_coords = compute_ref_plucker_coordinate(
        ray_batch['ray_o'], ray_batch['ray_d']
    )
    src_rays_coords = compute_src_plucker_coordinate(
        pts_ref, ray_batch['static_src_cameras']
    )

    raw_coarse_static = model.net_coarse_st(
        pts_ref,
        ref_rays_coords,
        src_rays_coords,
        rgb_feat_static,
        input_ray_dir,
        ray_diff_static,
        mask_static,
    )  # [N_rays, N_samples, 4]

    outputs_coarse_ref = raw2outputs(
        raw_coarse_ref,
        raw_coarse_static,
        z_vals,
        pixel_mask_ref,
        pixel_mask_static,
    )

    ret['outputs_coarse_ref'] = outputs_coarse_ref

  # ============================== END OF COARSE NETWORK
  assert N_importance > 0

  # detach since we would like to decouple the coarse and fine networks
  weights = (
      outputs_coarse_ref['weights'].clone().detach()
  )  # [N_rays, N_samples]
  if inv_uniform:
    inv_z_vals = 1.0 / z_vals
    inv_z_vals_mid = 0.5 * (
        inv_z_vals[:, 1:] + inv_z_vals[:, :-1]
    )  # [N_rays, N_samples-1]
    weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
    inv_z_vals = sample_pdf(
        bins=torch.flip(inv_z_vals_mid, dims=[1]),
        weights=torch.flip(weights, dims=[1]),
        N_samples=N_importance,
        det=det,
    )  # [N_rays, N_importance]
    z_samples = 1.0 / inv_z_vals
  else:
    # take mid-points of depth samples
    z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N_rays, N_samples-1]
    weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
    z_samples = sample_pdf(
        bins=z_vals_mid, weights=weights, N_samples=N_importance, det=det
    )  # [N_rays, N_importance]

  z_vals = torch.cat(
      (z_vals, z_samples), dim=-1
  )  # [N_rays, N_samples + N_importance]

  # samples are sorted with increasing depth
  z_vals, _ = torch.sort(z_vals, dim=-1)
  N_total_samples = N_samples + N_importance

  # compute normalized distance from z, for distortion loss
  near_depth_value = ray_batch['depth_range'][0, 0]
  far_depth_value = ray_batch['depth_range'][0, 1]
  s_vals = z_to_s(z_vals, near_depth_value, far_depth_value)

  viewdirs = ray_batch['ray_d'].unsqueeze(1).repeat(1, N_total_samples, 1)
  ray_o = ray_batch['ray_o'].unsqueeze(1).repeat(1, N_total_samples, 1)
  pts_ref_fine = (
      z_vals.unsqueeze(2) * viewdirs + ray_o
  )  # [N_rays, N_samples + N_importance, 3]

  # Rendeirng with both coarse and fine samples
  (
      outputs_fine_ref,
      outputs_fine_ref_dy,
      outputs_fine_anchor,
      outputs_fine_anchor_dy,
  ) = fine_render_rays(
      projector=projector,
      ray_batch=ray_batch,
      featmaps=fine_featmaps,
      pts_ref=pts_ref_fine,
      z_vals=z_vals,
      s_vals=s_vals,
      ref_time_embedding=ref_time_embedding,
      anchor_time_embedding=anchor_time_embedding,
      ref_frame_idx=ref_frame_idx,
      anchor_frame_idx=anchor_frame_idx,
      ref_time_offset=ref_time_offset,
      anchor_time_offset=anchor_time_offset,
      net_dy=model.net_fine_dy,
      net_st=model.net_fine_st,
      motion_mlp=model.motion_mlp_fine,
      trajectory_basis=model.trajectory_basis_fine,
      occ_weights_mode=args.occ_weights_mode,
      is_train=is_train,
  )

  ret['outputs_fine_ref'] = outputs_fine_ref
  ret['outputs_fine_ref_dy'] = outputs_fine_ref_dy

  # ======================== end for consistency
  ret['outputs_fine_anchor'] = outputs_fine_anchor
  ret['outputs_fine_anchor_dy'] = outputs_fine_anchor_dy

  return ret


def render_rays_mono(
    frame_idx,
    time_embedding,
    time_offset,
    ray_batch,
    model,
    featmaps,
    projector,
    N_samples,
    args,
    inv_uniform=False,
    N_importance=0,
    raw_noise_std=0.0,
    det=False,
    white_bkgd=False,
    is_train=True,
    num_vv=2,
):
  """Function for rendering rays for model for monocular videos.

  Args:
    frame_idx: reference video frame index
    time_embedding: reference time embedding
    time_offset: offset w.r.t reference time
    ray_batch: input ray information
    model: dynibar model
    projector: perspective projection module
    coarse_featmaps: coarse model 2D feature maps
    fine_featmaps: fine model 2D feature maps
    N_samples: number of coarse samples
    args: input argument list
    inv_uniform: disparity sampling
    N_importance: number of fine samples
    raw_noise_std: noise standard devition added to density output
    det: deterministic sampling
    white_bkgd: having white background?
    is_train: is training or not?
    num_vv: number virtual source views

  Returns:
    outputs_coarse_anchor: coarse composited model output from nearby time
    outputs_coarse_anchor_dy: coarse dynamic model output from nearby time
    outputs_coarse_ref: coarse composited model output from reference time
    outputs_coarse_ref_dy: coarse dynamic model output from reference time
    outputs_coarse_st: coarse static model output from reference time

  """
  # weights used in cross-time rendering.
  #   0: rendering by compositing both models if time interval == 1,
  #   otherwise from dynamic model only.
  #   1: rendering from dynamic model only.
  #   2: rendering by compositing both models for all time intervals
  occ_weights_mode = (
      args.occ_weights_mode
  )

  ref_frame_idx, anchor_frame_idx = frame_idx[0], frame_idx[1]
  ref_time_embedding, anchor_time_embedding = (
      time_embedding[0],
      time_embedding[1],
  )
  ref_time_offset, anchor_time_offset = time_offset[0], time_offset[1]
  num_frames = int(ref_frame_idx / ref_time_embedding)

  input_ray_dir = F.normalize(ray_batch['ray_d'], dim=-1)

  ret = {'outputs_coarse': None, 'outputs_fine': None}

  # pts: [N_rays, N_samples, 3]
  # z_vals: [N_rays, N_samples]
  pts_ref, z_vals, s_vals = sample_along_camera_ray(
      ray_o=ray_batch['ray_o'],
      ray_d=ray_batch['ray_d'],
      depth_range=ray_batch['depth_range'],
      N_samples=N_samples,
      inv_uniform=inv_uniform,
      det=det,
  )
  N_rays, N_samples = pts_ref.shape[:2]
  num_last_samples = int(round(N_samples * 0.1))

  # Scene motion at reference time in global coordinate system
  ref_time_embedding_ = ref_time_embedding[None, None, :].repeat(
      N_rays, N_samples, 1
  )

  ref_xyzt = (
      torch.cat([pts_ref, ref_time_embedding_], dim=-1)
      .float()
      .to(pts_ref.device)
  )
  raw_coeff_xyz = model.motion_mlp(ref_xyzt)
  raw_coeff_xyz[:, -num_last_samples:, :] *= 0.0

  num_basis = model.trajectory_basis.shape[1]
  raw_coeff_x = raw_coeff_xyz[..., 0:num_basis]
  raw_coeff_y = raw_coeff_xyz[..., num_basis : num_basis * 2]
  raw_coeff_z = raw_coeff_xyz[..., num_basis * 2 : num_basis * 3]

  ref_traj_pts_dict = {}
  # always use 6 nearby source views for dynamic model.
  for offset in [-3, -2, -1, 0, 1, 2, 3]:  
    traj_pts_ref = compute_traj_pts(
        raw_coeff_x,
        raw_coeff_y,
        raw_coeff_z,
        model.trajectory_basis[None, None, ref_frame_idx + offset, :],
    )

    ref_traj_pts_dict[offset] = traj_pts_ref

  pts_3d_seq_ref = []
  for offset in ref_time_offset:
    pts_3d_seq_ref.append(
        pts_ref + (ref_traj_pts_dict[offset] - ref_traj_pts_dict[0])
    )

  # adding src virtual views
  for _ in range(num_vv):
    pts_3d_seq_ref.append(pts_ref)

  pts_3d_seq_ref = torch.stack(pts_3d_seq_ref, 0)
  pts_3d_static = pts_ref[None, ...].repeat(
      ray_batch['static_src_rgbs'].shape[1], 1, 1, 1
  )

  # Feature query from source view with scene motion,
  # rendering at target view and for dyanmic model
  rgb_feat_ref, ray_diff_ref, mask_ref = projector.compute_with_motions(
      pts_ref,
      pts_3d_seq_ref,
      ray_batch['camera'],
      ray_batch['src_rgbs'],
      ray_batch['src_cameras'],
      featmaps=featmaps[0],
  )  # [N_rays, N_samples, N_views, x]

  # Feature query from source view without scene motion,
  # rendering at target view and for static model
  rgb_feat_static, ray_diff_static, mask_static = (
      projector.compute_with_motions(
          pts_ref,
          pts_3d_static,
          ray_batch['camera'],
          ray_batch['static_src_rgbs'],
          ray_batch['static_src_cameras'],
          featmaps=featmaps[2],
      )
  )  # [N_rays, N_samples, N_views, x]

  pixel_mask_ref = (
      mask_ref[..., 0].sum(dim=2) > 1
  )  # [N_rays, N_samples], should at least have 2 observations
  pixel_mask_static = (
      mask_static[..., 0].sum(dim=2) > 1
  )  # [N_rays, N_samples], should at least have 2 observations

  ref_time_diff = torch.from_numpy(np.array(ref_time_offset)).to(
      rgb_feat_ref.device
  )
  ref_time_diff = ref_time_diff[None, None, :, None].expand(
      ray_diff_ref.shape[0], ray_diff_ref.shape[1], -1, -1
  )

  raw_coarse_ref = model.net_coarse_dy(
      pts_ref,
      rgb_feat_ref,
      input_ray_dir,
      ray_diff_ref,
      ref_time_diff,
      mask_ref,
      ref_time_embedding_,
  )  # [N_rays, N_samples, 4]

  ref_rays_coords = compute_ref_plucker_coordinate(
      ray_batch['ray_o'], ray_batch['ray_d']
  )
  src_rays_coords = compute_src_plucker_coordinate(
      pts_ref, ray_batch['static_src_cameras']
  )

  raw_coarse_static = model.net_coarse_st(
      pts_ref,
      ref_rays_coords,
      src_rays_coords,
      rgb_feat_static,
      input_ray_dir,
      ray_diff_static,
      mask_static,
  )  # [N_rays, N_samples, 4]

  outputs_coarse_ref = raw2outputs(
      raw_coarse_ref,
      raw_coarse_static,
      z_vals,
      pixel_mask_ref,
      pixel_mask_static,
  )

  outputs_coarse_st = raw2outputs_vanilla(
      raw_coarse_static, z_vals, pixel_mask_static
  )

  outputs_coarse_ref_dy = raw2outputs_vanilla(
      raw_coarse_ref, z_vals, pixel_mask_ref
  )

  render_flows = compute_optical_flow(
      outputs_coarse_ref,
      pts_3d_seq_ref[:6],
      ray_batch['src_cameras'][:, :6, ...],
      ray_batch['uv_grid'],
  )
  outputs_coarse_ref['render_flows'] = render_flows
  outputs_coarse_ref['s_vals'] = s_vals

  exp_sf_p1 = torch.sum(
      outputs_coarse_ref['weights'][..., None]
      * (ref_traj_pts_dict[1] - ref_traj_pts_dict[0]),
      dim=-2,
  )
  exp_sf_m1 = torch.sum(
      outputs_coarse_ref['weights'][..., None]
      * (ref_traj_pts_dict[-1] - ref_traj_pts_dict[0]),
      dim=-2,
  )
  outputs_coarse_ref['exp_sf'] = torch.max(exp_sf_p1, exp_sf_m1).detach()

  # Cross-time rendering for temporal consistency
  if is_train:
    # First we compute scene flows
    sf_seq = []
    for offset in [-2, -1, 0, 1, 2, 3]:
      sf = ref_traj_pts_dict[offset] - ref_traj_pts_dict[offset - 1]
      sf_seq.append(sf)
    sf_seq = torch.stack(sf_seq, 0)

    # This part is computed for cycle consistency loss.
    # pts_anchor is nearby selected points for cross-time rendering.
    pts_anchor = pts_ref + (
        ref_traj_pts_dict[anchor_frame_idx - ref_frame_idx]
        - ref_traj_pts_dict[0]
    )
    anchor_time_embedding_ = (
        anchor_time_embedding[None, None, :]
        .repeat(N_rays, N_samples, 1)
        .float()
    )

    xyzt_anchor = (
        torch.cat([pts_anchor, anchor_time_embedding_], dim=-1)
        .float()
        .to(pts_ref.device)
    )

    # compute motion trajectory at displaced new points.
    raw_coeff_xyz_anchor = model.motion_mlp(xyzt_anchor)
    raw_coeff_xyz_anchor[:, -num_last_samples:, :] *= 0.0

    raw_coeff_x_anchor = raw_coeff_xyz_anchor[..., 0:num_basis]
    raw_coeff_y_anchor = raw_coeff_xyz_anchor[..., num_basis : num_basis * 2]
    raw_coeff_z_anchor = raw_coeff_xyz_anchor[
        ..., num_basis * 2 : num_basis * 3
    ]

    traj_pts_anchor_0 = compute_traj_pts(
        raw_coeff_x_anchor,
        raw_coeff_y_anchor,
        raw_coeff_z_anchor,
        model.trajectory_basis[None, None, anchor_frame_idx, :],
    )

    pts_3d_seq_anchor = []
    pts_traj_ref = []
    pts_traj_anchor = []

    # Compute motion trajectory of point at referece time, and its
    # corresponding points at nearby selected time,
    # and match their orders in the two lists.
    for offset in anchor_time_offset:
      ref_offset = anchor_frame_idx + offset - ref_frame_idx

      traj_pts_anchor = compute_traj_pts(
          raw_coeff_x_anchor,
          raw_coeff_y_anchor,
          raw_coeff_z_anchor,
          model.trajectory_basis[None, None, anchor_frame_idx + offset, :],
      )

      temp_pts = pts_anchor + (traj_pts_anchor - traj_pts_anchor_0)
      pts_3d_seq_anchor.append(temp_pts)

      if ref_offset not in ref_traj_pts_dict:
        continue

      pts_traj_anchor.append(temp_pts)
      pts_traj_ref.append(
          pts_ref + ref_traj_pts_dict[ref_offset] - ref_traj_pts_dict[0]
      )

    # Adding src virtual views for nearby selected views.
    for _ in range(num_vv):
      pts_3d_seq_anchor.append(pts_anchor)

    pts_traj_ref = torch.stack(pts_traj_ref, 0)
    pts_traj_anchor = torch.stack(pts_traj_anchor, 0)
    pts_3d_seq_anchor = torch.stack(pts_3d_seq_anchor, 0)

    # Feature query from source view with scene motion
    # w.r.t nearby selected time
    rgb_feat_anchor, ray_diff_anchor, mask_anchor = (
        projector.compute_with_motions(
            pts_ref,
            pts_3d_seq_anchor,
            ray_batch['camera'],
            ray_batch['anchor_src_rgbs'],
            ray_batch['anchor_src_cameras'],
            featmaps=featmaps[1],
        )
    )  # [N_rays, N_samples, N_views, x]

    anchor_time_diff = torch.from_numpy(np.array(anchor_time_offset)).to(
        rgb_feat_anchor.device
    )
    anchor_time_diff = anchor_time_diff[None, None, :, None].expand(
        ray_diff_anchor.shape[0], ray_diff_anchor.shape[1], -1, -1
    )

    pixel_mask_anchor = (
        mask_anchor[..., 0].sum(dim=2) > 0
    )  # [N_rays, N_samples], should at least have 2 observations
    raw_coarse_anchor = model.net_coarse_dy(
        pts_anchor,
        rgb_feat_anchor,
        input_ray_dir,
        ray_diff_anchor,
        anchor_time_diff,
        mask_anchor,
        anchor_time_embedding_,
    )  # [N_rays, N_samples, 4]
    # Obtain per-sample color and density at nearby selected time,
    # composited from dynamic and static models.
    outputs_coarse_anchor = raw2outputs(
        raw_coarse_anchor,
        raw_coarse_static,
        z_vals,
        pixel_mask_anchor,
        pixel_mask_static,
    )
    # Obtain per-sample color and density at nearby selected time,
    # from dynamic model only.
    outputs_coarse_anchor_dy = raw2outputs_vanilla(
        raw_coarse_anchor, z_vals, pixel_mask_anchor
    )
    occ_score_dy = (
        outputs_coarse_ref_dy['weights'] - outputs_coarse_anchor_dy['weights']
    )
    occ_score_dy = occ_score_dy.detach()
    occ_weights_dy = 1.0 - torch.abs(occ_score_dy)
    occ_weight_dy_map = 1.0 - torch.abs(torch.sum(occ_score_dy, dim=1))

    # compute disocclusion weights for cross-time rendering.
    if occ_weights_mode == 0:  # mix-mode
      time_diff = np.abs(ref_frame_idx - anchor_frame_idx)
      if time_diff > 1:  # composite-dy
        occ_score = (
            outputs_coarse_ref['weights_dy']
            - outputs_coarse_anchor['weights_dy']
        )
      else:  # full
        occ_score = (
            outputs_coarse_ref['weights'] - outputs_coarse_anchor['weights']
        )
    elif occ_weights_mode == 1:  # composite-dy
      occ_score = (
          outputs_coarse_ref['weights_dy'] - outputs_coarse_anchor['weights_dy']
      )
    elif occ_weights_mode == 2:  # full
      occ_score = (
          outputs_coarse_ref['weights'] - outputs_coarse_anchor['weights']
      )
    else:
      raise NotImplementedError

    occ_score = occ_score.detach()

    occ_weights = 1.0 - torch.abs(occ_score)
    occ_weight_map = 1.0 - torch.abs(torch.sum(occ_score, dim=1))

    outputs_coarse_anchor['occ_weights'] = occ_weights
    outputs_coarse_anchor['occ_weight_map'] = occ_weight_map

    outputs_coarse_anchor['pts_traj_ref'] = pts_traj_ref
    outputs_coarse_anchor['pts_traj_anchor'] = pts_traj_anchor
    outputs_coarse_anchor['sf_seq'] = sf_seq

    outputs_coarse_anchor_dy['occ_weights'] = occ_weights_dy
    outputs_coarse_anchor_dy['occ_weight_map'] = occ_weight_dy_map

    ret['outputs_coarse_anchor'] = outputs_coarse_anchor
    ret['outputs_coarse_anchor_dy'] = outputs_coarse_anchor_dy

  # ======================== end for cross-time consistency
  ret['outputs_coarse_ref'] = outputs_coarse_ref
  ret['outputs_coarse_ref_dy'] = outputs_coarse_ref_dy
  ret['outputs_coarse_st'] = outputs_coarse_st

  return ret
