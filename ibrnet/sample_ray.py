"""Utility class for sampling data corresponding to rays from images."""

import numpy as np
import torch
import torch.nn.functional as F
from kornia import create_meshgrid

rng = np.random.RandomState(234)


def parse_camera(params):
  H = params[:, 0]
  W = params[:, 1]
  intrinsics = params[:, 2:18].reshape((-1, 4, 4))
  c2w = params[:, 18:34].reshape((-1, 4, 4))
  return W, H, intrinsics, c2w


class RaySamplerSingleImage(object):
  """Sampling data corresponding to the rays from a target view.

  This class stores and returns following items at sampled pixel locations
  for training Dynibar:
    ray_o: ray origin at target view
    ray_d: ray direction at target view
    depth_range: scene depth bounds at target view
    camera: reference time camera parameters
    render_camera: rendered target view camera parameters
    anchor_camera: camera parameters for input view at nearby cross time
    rgb: image at reference time
    src_rgbs: source view images w.r.t reference time for dynamic model
    src_cameras: source view camera parameters w.r.t reference time for
    dynamic model
    anchor_src_rgbs: source view images w.r.t nearby cross time for dynamic
    model.
    anchor_src_cameras: source view camera parameters w.r.t 
    nearby cross time for dynamic model.
    static_src_rgbs: source view images for static model
    static_src_cameras: source view camera parameters for static model
    static_src_masks: dynamic masks of source views for static model
    disp: disparity map
    motion_mask: dynamic mask
    static_mask: static masks
    uv_grid: 2D pixel coorindate in image space
    flows: observed 2D optical flows
    masks: optical flow vailid masks
  """

  def __init__(self, data, device, resize_factor=1, render_stride=1):
    super().__init__()
    self.render_stride = render_stride
    self.rgb = data['rgb'] if 'rgb' in data.keys() else None
    self.disp = data['disp'] if 'disp' in data.keys() else None

    self.motion_mask = (
        data['motion_mask'] if 'motion_mask' in data.keys() else None
    )

    self.static_mask = (
        data['static_mask'] if 'static_mask' in data.keys() else None
    )

    self.flows = data['flows'].squeeze(0) if 'flows' in data.keys() else None
    self.masks = data['masks'].squeeze(0) if 'masks' in data.keys() else None

    self.camera = data['camera']
    self.render_camera = (
        data['render_camera'] if 'render_camera' in data.keys() else None
    )

    self.anchor_camera = (
        data['anchor_camera'] if 'anchor_camera' in data.keys() else None
    )
    self.rgb_path = data['rgb_path']
    self.depth_range = data['depth_range']
    self.device = device
    W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)

    self.batch_size = len(self.camera)

    self.H = int(H[0])
    self.W = int(W[0])
    self.uv_grid = create_meshgrid(
        self.H, self.W, normalized_coordinates=False
    )[0].to(
        self.device
    )  # (H, W, 2)

    self.rays_o, self.rays_d = self.get_rays_single_image(
        self.H, self.W, self.intrinsics, self.c2w_mat
    )

    if self.rgb is not None:
      self.rgb = self.rgb.reshape(-1, 3)

    if self.disp is not None:
      self.disp = self.disp.reshape(-1, 1)

    if self.motion_mask is not None:
      self.motion_mask = self.motion_mask.reshape(-1, 1)

    if self.static_mask is not None:
      self.static_mask = self.static_mask.reshape(-1, 1)

    if self.flows is not None:
      self.flows = self.flows.reshape(self.flows.shape[0], -1, 2)
      self.masks = self.masks.reshape(self.masks.shape[0], -1, 1)

    self.uv_grid = self.uv_grid.reshape(-1, 2)

    if 'src_rgbs' in data.keys():
      self.src_rgbs = data['src_rgbs']
    else:
      self.src_rgbs = None

    if 'src_cameras' in data.keys():
      self.src_cameras = data['src_cameras']
    else:
      self.src_cameras = None

    self.anchor_src_rgbs = (
        data['anchor_src_rgbs'] if 'anchor_src_rgbs' in data.keys() else None
    )
    self.anchor_src_cameras = (
        data['anchor_src_cameras']
        if 'anchor_src_cameras' in data.keys()
        else None
    )

    self.static_src_rgbs = (
        data['static_src_rgbs'] if 'static_src_rgbs' in data.keys() else None
    )
    self.static_src_cameras = (
        data['static_src_cameras']
        if 'static_src_cameras' in data.keys()
        else None
    )
    self.static_src_masks = (
        data['static_src_masks'] if 'static_src_masks' in data.keys() else None
    )


  def get_rays_single_image(self, H, W, intrinsics, c2w):
    """Return ray parameters (origin, direction) from a target view."""
    u, v = np.meshgrid(
        np.arange(W)[:: self.render_stride], np.arange(H)[:: self.render_stride]
    )
    u = u.reshape(-1).astype(dtype=np.float32)
    v = v.reshape(-1).astype(dtype=np.float32)
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
    pixels = torch.from_numpy(pixels)
    batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

    rays_d = (
        c2w[:, :3, :3]
        .bmm(torch.inverse(intrinsics[:, :3, :3]))
        .bmm(batched_pixels)
    ).transpose(1, 2)
    rays_d = rays_d.reshape(-1, 3)
    rays_o = (
        c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)
    )  # B x HW x 3
    return rays_o, rays_d

  def get_all(self):
    """Return all camera and ray information from a target view."""
    ret = {
        'ray_o': self.rays_o.to(self.device),
        'ray_d': self.rays_d.to(self.device),
        'depth_range': self.depth_range.to(self.device),
        'camera': self.camera.to(self.device),
        'render_camera': (
            self.render_camera.to(self.device)
            if self.render_camera is not None
            else None
        ),
        'anchor_camera': (
            self.anchor_camera.to(self.device)
            if self.anchor_camera is not None
            else None
        ),
        'rgb': self.rgb.to(self.device) if self.rgb is not None else None,
        'src_rgbs': (
            self.src_rgbs.to(self.device) if self.src_rgbs is not None else None
        ),
        'src_cameras': (
            self.src_cameras.to(self.device)
            if self.src_cameras is not None
            else None
        ),
        'anchor_src_rgbs': (
            self.anchor_src_rgbs.to(self.device)
            if self.anchor_src_rgbs is not None
            else None
        ),
        'anchor_src_cameras': (
            self.anchor_src_cameras.to(self.device)
            if self.anchor_src_cameras is not None
            else None
        ),
        'static_src_rgbs': (
            self.static_src_rgbs.to(self.device)
            if self.static_src_rgbs is not None
            else None
        ),
        'static_src_cameras': (
            self.static_src_cameras.to(self.device)
            if self.static_src_cameras is not None
            else None
        ),
        'static_src_masks': (
            self.static_src_masks.to(self.device)
            if self.static_src_masks is not None
            else None
        ),
        'disp': (
            self.disp.to(self.device).squeeze()
            if self.disp is not None
            else None
        ),
        'motion_mask': (
            self.motion_mask.to(self.device).squeeze()
            if self.motion_mask is not None
            else None
        ),
        'static_mask': (
            self.static_mask.to(self.device).squeeze()
            if self.static_mask is not None
            else None
        ),
        'uv_grid': self.uv_grid.to(self.device),
        'flows': self.flows.to(self.device) if self.flows is not None else None,
        'masks': self.masks.to(self.device) if self.masks is not None else None,
    }
    return ret

  def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
    """Sample pixel randomly from the target view."""
    if sample_mode == 'center':
      border_H = int(self.H * (1 - center_ratio) / 2.0)
      border_W = int(self.W * (1 - center_ratio) / 2.0)

      # pixel coordinates
      u, v = np.meshgrid(
          np.arange(border_H, self.H - border_H),
          np.arange(border_W, self.W - border_W),
      )
      u = u.reshape(-1)
      v = v.reshape(-1)

      select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
      select_inds = v[select_inds] + self.W * u[select_inds]

    elif sample_mode == 'uniform':
      # Random from one image
      select_inds = rng.choice(self.H*self.W, size=(N_rand,), replace=False)
    else:
      raise NotImplementedError

    return select_inds

  def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
    """Randomly sample pixel and pixel data from the target view."""
    select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)

    rays_o = self.rays_o[select_inds]
    rays_d = self.rays_d[select_inds]

    if self.rgb is not None:
      rgb = self.rgb[select_inds]
      disp = self.disp[select_inds].squeeze()
      motion_mask = self.motion_mask[select_inds].squeeze()
      static_mask = self.static_mask[select_inds].squeeze()

      flows = self.flows[:, select_inds, :]
      masks = self.masks[:, select_inds, :]

      uv_grid = self.uv_grid[select_inds]

    else:
      raise NotImplementedError

    ret = {
        'ray_o': rays_o.to(self.device),
        'ray_d': rays_d.to(self.device),
        'camera': self.camera.to(self.device),
        'anchor_camera': self.anchor_camera.to(self.device),
        'depth_range': self.depth_range.to(self.device),
        'rgb': rgb.to(self.device) if rgb is not None else None,
        'disp': disp.to(self.device),
        'motion_mask': motion_mask.to(self.device),
        'static_mask': static_mask.to(self.device),
        'uv_grid': uv_grid.to(self.device),
        'flows': flows.to(self.device),
        'masks': masks.to(self.device),
        'src_rgbs': (
            self.src_rgbs.to(self.device) if self.src_rgbs is not None else None
        ),
        'src_cameras': (
            self.src_cameras.to(self.device)
            if self.src_cameras is not None
            else None
        ),
        'static_src_rgbs': (
            self.static_src_rgbs.to(self.device)
            if self.static_src_rgbs is not None
            else None
        ),
        'static_src_cameras': (
            self.static_src_cameras.to(self.device)
            if self.static_src_cameras is not None
            else None
        ),
        'static_src_masks': (
            self.static_src_masks.to(self.device)
            if self.static_src_masks is not None
            else None
        ),
        'anchor_src_rgbs': (
            self.anchor_src_rgbs.to(self.device)
            if self.anchor_src_rgbs is not None
            else None
        ),
        'anchor_src_cameras': (
            self.anchor_src_cameras.to(self.device)
            if self.anchor_src_cameras is not None
            else None
        ),
        'selected_inds': select_inds,
    }
    return ret
