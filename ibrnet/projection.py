"""Class definition for perspective projection."""

import torch
import torch.nn.functional as F


class Projector:
  """Class for performing perspective projection."""

  def __init__(self, device):
    self.device = device

  def inbound(self, pixel_locations, h, w):
    """Check if the pixel locations are in valid range."""
    return (
        (pixel_locations[..., 0] <= w - 1.0)
        & (pixel_locations[..., 0] >= 0)
        & (pixel_locations[..., 1] <= h - 1.0)
        & (pixel_locations[..., 1] >= 0)
    )

  def normalize(self, pixel_locations, h, w):
    """Normalize pixel locations for grid_sampler function."""
    resize_factor = torch.tensor([w - 1.0, h - 1.0]).to(self.device)[
        None, None, :
    ]
    normalized_pixel_locations = (
        2 * pixel_locations / resize_factor - 1.0
    )  # [n_views, n_points, 2]
    return normalized_pixel_locations

  def compute_projections(self, xyz, train_cameras):
    """Project 3D points into views using training camera parameteres."""
    original_shape = xyz.shape[:-1]
    xyz = xyz.reshape(original_shape[0], -1, 3)

    num_views = len(train_cameras)
    train_intrinsics = train_cameras[:, 2:18].reshape(
        -1, 4, 4
    )  # [n_views, 4, 4]
    train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
    xyz_h = torch.cat(
        [xyz, torch.ones_like(xyz[..., :1])], dim=-1
    )  # [n_points, 4]

    projections = train_intrinsics.bmm(torch.inverse(train_poses)).bmm(
        xyz_h.permute(0, 2, 1)
    )  # [n_views, 4, n_points]

    projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
    pixel_locations = projections[..., :2] / torch.clamp(
        projections[..., 2:3], min=1e-8
    )  # [n_views, n_points, 2]
    pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)

    mask = projections[..., 2] > 0  # a point is invalid if behind the camera
    return pixel_locations.reshape(
        (num_views,) + original_shape[1:] + (2,)
    ), mask.reshape((num_views,) + original_shape[1:])

  def compute_angle(self, xyz_st, xyz, query_camera, train_cameras):
    """Compute difference of viewing angle between rays from source and ones from target view.
    
    Args:

      xyz_st: reference 3D point location without scene motion
      xyz: 3D positions displaced by scene motion at nearby times
      query_camera: target view camera parameters
      train_imgs: source view images

    Returns:
      Difference of viewing angle between rays from source and ones from target
      view.
    """
    original_shape = xyz.shape[:-1]
    xyz_st_ = xyz_st.reshape(xyz_st.shape[0], -1, 3)
    xyz_ = xyz.reshape(xyz.shape[0], -1, 3)

    train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
    num_views = len(train_poses)
    query_pose = (
        query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)
    )  # [n_views, 4, 4]

    ray2tar_pose = F.normalize(
        query_pose[:, :3, 3].unsqueeze(1) - xyz_st_, dim=-1
    )
    ray2train_pose = F.normalize(
        train_poses[:, :3, 3].unsqueeze(1) - xyz_, dim=-1
    )
    ray_diff = ray2tar_pose - ray2train_pose

    ray_diff_dot = torch.sum(
        ray2tar_pose * ray2train_pose, dim=-1, keepdim=True
    )
    ray_diff_direction = F.normalize(
        ray_diff, dim=-1
    )  # ray_diff / torch.clamp(ray_diff_norm, min=1e-6)

    ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
    return ray_diff.reshape((num_views,) + original_shape[1:] + (4,))

  def compute_with_motions(
      self, xyz_st, xyz, query_camera, train_imgs, train_cameras, featmaps
  ):
    """Extract 2D feature by projecting 3D points displaced by scene motion.

    Args:
      xyz_st: reference point location without scene motion
      xyz: 3D point positions displaced by scene motion
      query_camera: target view camera parameters
      train_imgs: source view images
      train_cameras: source view camera parameters
      featmaps: source view 2D image feature maps.

    Returns:
      rgb_feat_sampled: extracted 2D feature
      ray_diff: viewing angle difference between target ray and source ray
      mask: valid masks
    """

    assert (
        (train_imgs.shape[0] == 1)
        and (train_cameras.shape[0] == 1)
        and (query_camera.shape[0] == 1)
    ), 'only support batch_size=1 for now'

    xyz_st = xyz_st[None, ...].expand(xyz.shape[0], -1, -1, -1)

    train_imgs = train_imgs.squeeze(0)  # [n_views, h, w, 3]
    train_cameras = train_cameras.squeeze(0)  # [n_views, 34]
    query_camera = query_camera.squeeze(0)  # [34, ]

    train_imgs = train_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

    h, w = train_cameras[0][:2]

    # compute the projection of the query points to each reference image
    pixel_locations, mask_in_front = self.compute_projections(
        xyz, train_cameras
    )

    normalized_pixel_locations = self.normalize(
        pixel_locations, h, w
    )  # [n_views, n_rays, n_samples, 2]

    # rgb sampling
    rgbs_sampled = F.grid_sample(
        train_imgs, normalized_pixel_locations, align_corners=True
    )
    rgbs_sampled_ = rgbs_sampled.permute(
        2, 3, 0, 1
    )  # [n_rays, n_samples, n_views, 3]

    # deep feature sampling
    feat_sampled = F.grid_sample(
        featmaps, normalized_pixel_locations, align_corners=True
    )
    feat_sampled = feat_sampled.permute(
        2, 3, 0, 1
    )  # [n_rays, n_samples, n_views, d]
    rgb_feat_sampled = torch.cat(
        [rgbs_sampled_, feat_sampled], dim=-1
    )  # [n_rays, n_samples, n_views, d+3]

    inbound = self.inbound(pixel_locations, h, w)
    ray_diff = self.compute_angle(
        xyz_st, xyz, query_camera, train_cameras
    ).detach()

    ray_diff = ray_diff.permute(1, 2, 0, 3)
    mask = (
        (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]
    )  # [n_rays, n_samples, n_views, 1]

    return rgb_feat_sampled, ray_diff, mask
