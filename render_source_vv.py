"""Rendering virutal source views from video depth, used for monocular video."""

import argparse
import glob
import os

import cv2
import imageio.v2 as imageio
import kornia
import numpy as np
import skimage.morphology
from splatting import splatting_function
import torch

def render_forward_splat(src_imgs, src_depths, r_cam, t_cam, k_src, k_dst):
  '''Point cloud rendering from RGBD images.'''
  batch_size = src_imgs.shape[0]

  rot = r_cam
  t = t_cam
  k_src_inv = k_src.inverse()

  x = np.arange(src_imgs[0].shape[1])
  y = np.arange(src_imgs[0].shape[0])
  coord = np.stack(np.meshgrid(x, y), -1)
  coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)
  coord = coord.astype(np.float32)
  coord = torch.as_tensor(coord, dtype=k_src.dtype, device=k_src.device)
  coord = coord[None, ..., None].repeat(batch_size, 1, 1, 1, 1)

  depth = src_depths[:, :, :, None, None]

  # from reference to target viewpoint
  pts_3d_ref = depth * k_src_inv[:, None, None, ...] @ coord
  pts_3d_tgt = rot[:, None, None, ...] @ pts_3d_ref + t[:, None, None, :, None]
  points = k_dst[:, None, None, ...] @ pts_3d_tgt
  points = points.squeeze(-1)

  new_z = points[:, :, :, [2]].clone().permute(0, 3, 1, 2)  # b,1,h,w
  points = points / torch.clamp(points[:, :, :, [2]], 1e-8, None)

  src_ims_ = src_imgs.permute(0, 3, 1, 2)
  num_channels = src_ims_.shape[1]

  flow = points - coord.squeeze(-1)
  flow = flow.permute(0, 3, 1, 2)[:, :2, ...]

  importance = 1.0 / (new_z)
  importance_min = importance.amin((1, 2, 3), keepdim=True)
  importance_max = importance.amax((1, 2, 3), keepdim=True)
  weights = (importance - importance_min) / (
      importance_max - importance_min + 1e-6
  ) * 20 - 10
  src_mask_ = torch.ones_like(new_z)

  input_data = torch.cat([src_ims_, (1.0 / (new_z)), src_mask_], 1)

  output_data = splatting_function(
      'softmax', input_data.cuda(), flow.cuda(), weights.detach().cuda()
  )

  warp_feature = output_data[:, 0:num_channels, ...]
  warp_disp = output_data[:, num_channels : num_channels + 1, ...]
  # warp_mask = output_data[:, num_channels + 1 : num_channels + 2, ...]

  return warp_feature, warp_disp#, warp_mask

def render_wander_path(c2w, hwf, bd_scale, max_disp_=50, xyz=[1, 0, 1]):
  """Render nearby virtual source views with displacement in x and z direciton."""
  num_frames = 60
  max_disp = max_disp_ * bd_scale
  max_trans = (
      max_disp / hwf[2][0]
  )
  output_poses = []

  for i in range(num_frames):

    x_trans = max_trans * np.cos(
        2.0 * np.pi * float(i) / float(num_frames)
    )  * xyz[0]
    y_trans = max_trans * np.sin(
        2.0 * np.pi * float(i) / float(num_frames)
    ) * xyz[1]
    z_trans = max_trans * np.cos(
        2.0 * np.pi * float(i) / float(num_frames)
    ) * xyz[2]

    i_pose = np.concatenate(
        [
            np.concatenate(
                [
                    np.eye(3),
                    np.array([x_trans, y_trans, z_trans])[:, np.newaxis],
                ],
                axis=1,
            ),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
        ],
        axis=0,
    )

    i_pose = np.linalg.inv(
        i_pose
    )  # torch.tensor(np.linalg.inv(i_pose)).float()

    ref_pose = np.concatenate(
        [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
    )

    render_pose = np.dot(ref_pose, i_pose)

    output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))

  return np.array(output_poses + output_poses), num_frames


def sobel_fg_alpha(disp, mode='sobel', beta=10.0):
  """Create depth boundary mask."""
  sobel_grad = kornia.filters.spatial_gradient(
      disp, mode=mode, normalized=False
  )
  sobel_mag = torch.sqrt(
      sobel_grad[:, :, 0, ...] ** 2 + sobel_grad[:, :, 1, ...] ** 2
  )
  alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

  return alpha


FINAL_H = 288
USE_DPT = True

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument("--scene_name", type=str, 
                      # help='Scene name') # 'kid-running'
  parser.add_argument("--data_dir", type=str, 
                      help='data directory') # '/home/zhengqili/filestore/NSFF/nerf_data/release'
  parser.add_argument("--cvd_dir", type=str, 
                      help='video depth directory') # '/home/zhengqili/filestore/dynamic-video-DPT/monocular-results/kid-runningscene_flow_motion_field_shutterstock_epoch_15/epoch0015_test'

  args = parser.parse_args()

  data_path = os.path.join(
      args.data_dir, 'dense'
  )

  pt_out_list = sorted(
      glob.glob(
          os.path.join(
              args.cvd_dir,
              '*.npz',
          )
      )
  )

  try:
    original_img_path = os.path.join(data_path, 'images', '00000.png')
    o_img = imageio.imread(original_img_path)
  except:
    original_img_path = os.path.join(data_path, 'images', '00000.jpg')
    o_img = imageio.imread(original_img_path)

  o_ar = float(o_img.shape[1]) / float(o_img.shape[0])

  final_w, final_h = int(round(FINAL_H * o_ar)), int(FINAL_H)

  save_dir = os.path.join(
      data_path, 'source_virtual_views_%dx%d' % (final_w, final_h)
  )
  os.makedirs(save_dir, exist_ok=True)

  Ks = []
  mono_depths = []
  c2w_mats = []
  imgs = []
  bounds_mats = []
  points_cloud = []

  for i in range(0, len(pt_out_list)):
    pt_out_path = pt_out_list[i]
    out_name = pt_out_path.split('/')[-1]
    pt_data = np.load(pt_out_path)
    pred_depth = pt_data['depth'][0, 0, ...]
    cam_c2w = pt_data['cam_c2w'][0]
    img = pt_data['img_1'][0].transpose(1, 2, 0)

    c2w_mats.append(cam_c2w)
    bounds_mats.append(np.percentile(pred_depth, 5))
    K = pt_data['K'][0, 0, 0, ...].transpose()
    K[0, :] *= final_w / img.shape[1]
    K[1, :] *= final_h / img.shape[0]

  h, w, fx, fy = final_h, final_w, K[0, 0], K[1, 1]
  ff = (fx + fy) / 2.0
  # hwf = np.array([h, w, fx, fy]).reshape([1, 4])
  hwf = np.array([h, w, ff]).reshape([3, 1])

  c2w_mats = np.stack(c2w_mats, 0)
  bounds_mats = np.stack(bounds_mats, 0)

  bd_scale = bounds_mats.min() * 0.75

  poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])

  # must switch to [-y, x, z] from [x, -y, -z], NOT [r, u, -t]
  poses = np.concatenate(
      [poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :]],
      1,
  )
  poses = np.moveaxis(poses, -1, 0).astype(np.float32)

  num_samples = 4
  vv_poses_final = np.zeros((poses.shape[0], num_samples * 2, 3, 4))

  for ii in range(poses.shape[0]):
    print(ii)
    virtural_poses_0, num_render_0 = render_wander_path(
        poses[ii], hwf, bd_scale, 56 * 1.5, 
        xyz=[0., 1., 1.] # y, x, z
    )
    virtural_poses_1, num_render_1 = render_wander_path(
        poses[ii], hwf, bd_scale, 48 * 1.5, 
        xyz=[0.5, 1., 0.]
    )
    # this is for fixed viewpoint!
    start_idx = np.random.randint(0, num_render_0 // num_samples)

    vv_poses_final[ii, :num_samples, ...] = virtural_poses_0[
        5 : -1 : int(num_render_0 // num_samples)
    ][:num_samples, :3, :4]
    vv_poses_final[ii, num_samples:, ...] = virtural_poses_1[
        15 : -1 : int(num_render_1 // num_samples)
    ][:num_samples, :3, :4]

  np.save(
      os.path.join(data_path, 'source_vv_poses.npy'),
      np.moveaxis(vv_poses_final, 0, -1).astype(np.float32),
  )

  # switch back
  c2w_mats_vsv = np.concatenate(
      [
          vv_poses_final[..., 1:2],
          vv_poses_final[..., 0:1],
          -vv_poses_final[..., 2:3],
          vv_poses_final[..., 3:4],
      ],
      -1,
  )

  for i in range(0, len(pt_out_list)):
    save_sub_dir = os.path.join(save_dir, '%05d' % i)
    print(save_sub_dir)
    os.makedirs(save_sub_dir, exist_ok=True)
    pt_out_path = pt_out_list[i]

    out_name = pt_out_path.split('/')[-1]
    pt_data = np.load(pt_out_path)

    K = pt_data['K'][0, 0, 0, ...].transpose()
    img = pt_data['img_1'][0].transpose(1, 2, 0)
    cam_ref2w = pt_data['cam_c2w'][0]
    pred_depth = pt_data['depth'][0, 0, ...]
    pred_disp = 1.0 / pred_depth

    K[0, :] *= final_w / img.shape[1]
    K[1, :] *= final_h / img.shape[0]

    print('K ', K)
    assert abs(K[0, 0] - K[1, 1]) / abs(K[0, 0] + K[1, 1]) < 0.005

    pred_depth_ = cv2.resize(
        pred_depth, (final_w, final_h), interpolation=cv2.INTER_NEAREST
    )

    img = cv2.resize(img, (final_w, final_h), interpolation=cv2.INTER_AREA)
    pred_disp = cv2.resize(
        pred_disp, (final_w, final_h), interpolation=cv2.INTER_LINEAR
    )

    mode = 'sobel'
    beta = 0.5
    pred_depth = 1.0 / torch.from_numpy(pred_disp[None, None, ...])
    pred_depth = pred_depth / 10.0
    cur_alpha = sobel_fg_alpha(pred_depth, mode, beta=beta)[
        0, 0, ..., None
    ].numpy()

    for k in range(num_samples * 2):
      # render source view into target viewpoint
      rgba_pt = torch.from_numpy(
          np.concatenate(
              [np.array(img * 255.0), cur_alpha], axis=-1
          )
      )[None].float()
      disp_pt = torch.from_numpy(np.array(pred_disp))[
          None
      ].float()
      cam_tgt2w = np.eye(4)
      cam_tgt2w[:3, :4] = c2w_mats_vsv[i, k]
      T_ref2tgt = np.dot(np.linalg.inv(cam_tgt2w), cam_ref2w)

      fwd_rot = torch.from_numpy(T_ref2tgt[:3, :3])[None].float()
      fwd_t = torch.from_numpy(T_ref2tgt[:3, 3])[None].float()  # * metric_scale
      k_ref = torch.from_numpy(np.array(K))[None].float()

      render_rgba, render_depth = render_forward_splat(
          rgba_pt, 1.0 / disp_pt, fwd_rot, fwd_t, k_src=k_ref, k_dst=k_ref
      )

      render_rgb = np.clip(
          render_rgba[0, :3, ...].cpu().numpy().transpose(1, 2, 0) / 255.0,
          0.0,
          1.0,
      )
      mask = np.clip(
          render_rgba[0, 3:4, ...].cpu().numpy().transpose(1, 2, 0), 0.0, 1.0
      )
      mask = skimage.morphology.erosion(
          mask[..., 0] > 0.5, skimage.morphology.disk(1)
      )

      render_rgb_masked = render_rgb * mask[..., None]
      h, w = render_rgb_masked.shape[:2]
      imageio.imsave(
          os.path.join(save_sub_dir, '%02d.png' % k),
          np.uint8(255 * np.clip(render_rgb_masked, 0.0, 1.0)),
      )
