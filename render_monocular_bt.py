"""Script to render novel views from pretrained model."""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imageio.v2 as imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image_mono
from ibrnet.model import DynibarMono
from ibrnet.projection import Projector
from ibrnet.data_loaders.data_utils import get_nearest_pose_ids
from ibrnet.data_loaders.data_utils import get_interval_pose_ids
from ibrnet.data_loaders.llff_data_utils import load_mono_data
from ibrnet.data_loaders.llff_data_utils import batch_parse_llff_poses 
from ibrnet.data_loaders.llff_data_utils import batch_parse_vv_poses
import time
import os
import numpy as np
import cv2


class DynamicVideoDataset(Dataset):
  """Class for defining monocular video data.

  Attributes:
    folder_path: root path
    num_source_views: number of source views to sample
    mask_src_view: using mask to mask moving objects
    render_idx: rendering frame index
    max_range: max sampling frame range
    render_rgb_files: rendering RGB file path
    render_intrinsics: rendering camera intrinsics
    render_poses: rendering camera poses
    render_depth_range: rendering depth bounds
    h: image height
    w: image width
    train_intrinsics: training camera intrinisc
    train_poses: training camera poses
    train_rgb_files: training RGB path
    num_frames: number of video frames
    src_vv_c2w_mats: virtual views camera matrix
  """

  def __init__(self, args, scenes, **kwargs):
    self.folder_path = (
        args.folder_path
    )
    self.num_source_views = args.num_source_views
    self.mask_src_view = args.mask_src_view
    self.render_idx = args.render_idx
    self.max_range = args.max_range
    self.num_vv = args.num_vv
    print('num_source_views ', self.num_source_views)
    print('loading {} for rendering'.format(scenes))
    assert len(scenes) == 1

    scene = scenes[0]
    # for i, scene in enumerate(scenes):
    scene_path = os.path.join(self.folder_path, scene, 'dense')
    _, poses, src_vv_poses, bds, render_poses, _, rgb_files, _ = (
        load_mono_data(
            scene_path,
            height=args.training_height,
            render_idx=self.render_idx,
            load_imgs=False,
        )
    )
    near_depth = np.min(bds)

    if np.max(bds) < 10:
      far_depth = min(50, np.max(bds) + 15.0)
    else:
      far_depth = min(50, max(20, np.max(bds)))

    self.num_frames = len(rgb_files)

    intrinsics, c2w_mats = batch_parse_llff_poses(poses)
    h, w = poses[0][:2, -1]
    render_intrinsics, render_c2w_mats = batch_parse_llff_poses(render_poses)
    self.src_vv_c2w_mats = batch_parse_vv_poses(src_vv_poses)

    self.train_intrinsics = intrinsics 
    self.train_poses = c2w_mats 
    self.train_rgb_files = rgb_files
    
    self.render_intrinsics = render_intrinsics
    self.render_poses = render_c2w_mats 
    self.render_depth_range = [[near_depth, far_depth]] * self.num_frames
    self.h = [int(h)] * self.num_frames
    self.w = [int(w)] * self.num_frames

  def __len__(self):
    return len(self.render_poses)

  def __getitem__(self, idx):
    render_pose = self.render_poses[idx]
    intrinsics = self.render_intrinsics[idx]
    depth_range = self.render_depth_range[idx]

    train_rgb_files = self.train_rgb_files
    train_poses = self.train_poses
    train_intrinsics = self.train_intrinsics

    rgb_file = train_rgb_files[idx]
    rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0

    h, w = self.h[idx], self.w[idx]
    camera = np.concatenate(
        ([h, w], intrinsics.flatten(), render_pose.flatten())
    ).astype(np.float32)

    nearest_pose_ids = np.sort(
        [self.render_idx + offset for offset in [1, 2, 3, 0, -1, -2, -3]]
    )
    sp_pose_ids = get_nearest_pose_ids(
        render_pose, train_poses, tar_id=-1, angular_dist_method='dist'
    )

    static_pose_ids = []
    frame_interval = args.max_range // self.num_source_views
    interval_pose_ids = get_interval_pose_ids(
        render_pose,
        train_poses,
        tar_id=-1,
        angular_dist_method='dist',
        interval=frame_interval,
    )

    for sp_pose_id in interval_pose_ids:
      if len(static_pose_ids) >= (self.num_source_views * 2 + 1):
        break

      if np.abs(sp_pose_id - self.render_idx) > (
          self.max_range + self.num_source_views * 0.5
      ):
        continue

      static_pose_ids.append(sp_pose_id)

    static_pose_set = set(static_pose_ids)

    # if there is no sufficient src imgs, naively choose the closest images
    for sp_pose_id in sp_pose_ids[::5]:
      if len(static_pose_ids) >= (self.num_source_views * 2 + 1):
        break

      if sp_pose_id in static_pose_set:
        continue

      static_pose_ids.append(sp_pose_id)

    static_pose_ids = np.sort(static_pose_ids)

    assert len(static_pose_ids) == (self.num_source_views * 2 + 1)

    src_rgbs = []
    src_cameras = []
    for src_idx in nearest_pose_ids:
      src_rgb = (
          imageio.imread(train_rgb_files[src_idx]).astype(np.float32) / 255.0
      )
      train_pose = train_poses[src_idx]
      train_intrinsics_ = train_intrinsics[src_idx]
      src_rgbs.append(src_rgb)
      img_size = src_rgb.shape[:2]
      src_camera = np.concatenate(
          (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
      ).astype(np.float32)

      src_cameras.append(src_camera)

    # load src virtual views
    vv_pose_ids = get_nearest_pose_ids(
        render_pose,
        self.src_vv_c2w_mats[self.render_idx],
        tar_id=-1,
        angular_dist_method='dist',
    )

    # load virtual source views
    num_vv = self.num_vv
    for virtual_idx in vv_pose_ids[:num_vv]:
      src_vv_path = os.path.join(
          '/'.join(
              rgb_file.replace('images', 'source_virtual_views').split('/')[:-1]
          ),
          '%05d' % self.render_idx,
          '%02d.png' % virtual_idx,
      )
      src_rgb = imageio.imread(src_vv_path).astype(np.float32) / 255.0
      src_rgbs.append(src_rgb)
      img_size = src_rgb.shape[:2]

      src_camera = np.concatenate((
          list(img_size),
          intrinsics.flatten(),
          self.src_vv_c2w_mats[self.render_idx, virtual_idx].flatten(),
      )).astype(np.float32)

      src_cameras.append(src_camera)

    src_rgbs = np.stack(src_rgbs, axis=0)
    src_cameras = np.stack(src_cameras, axis=0)

    static_src_rgbs = []
    static_src_cameras = []
    # load src rgb for static view
    for st_near_id in static_pose_ids:
      src_rgb = (
          imageio.imread(train_rgb_files[st_near_id]).astype(np.float32) / 255.0
      )
      train_pose = train_poses[st_near_id]
      train_intrinsics_ = train_intrinsics[st_near_id]

      if self.mask_src_view:
        st_mask_path = os.path.join(
            '/'.join(rgb_file.split('/')[:-2]),
            'dynamic_masks',
            '%d.png' % st_near_id,
        )
        st_mask = imageio.imread(st_mask_path).astype(np.float32) / 255.0
        st_mask = cv2.resize(
            st_mask,
            (src_rgb.shape[1], src_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        if len(st_mask.shape) == 2:
          st_mask = st_mask[..., None]

        src_rgb = src_rgb * st_mask

      static_src_rgbs.append(src_rgb)
      img_size = src_rgb.shape[:2]
      src_camera = np.concatenate(
          (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
      ).astype(np.float32)

      static_src_cameras.append(src_camera)

    static_src_rgbs = np.stack(static_src_rgbs, axis=0)
    static_src_cameras = np.stack(static_src_cameras, axis=0)

    depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

    return {
        'camera': torch.from_numpy(camera),
        'rgb_path': '',
        'rgb': torch.from_numpy(rgb),
        'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
        'src_cameras': torch.from_numpy(src_cameras).float(),
        'static_src_rgbs': torch.from_numpy(static_src_rgbs[..., :3]).float(),
        'static_src_cameras': torch.from_numpy(static_src_cameras).float(),
        'depth_range': depth_range,
        'ref_time': float(self.render_idx / float(self.num_frames)),
        'id': self.render_idx,
        'nearest_pose_ids': nearest_pose_ids
        }

if __name__ == '__main__':
  parser = config_parser()
  args = parser.parse_args()
  args.distributed = False

  test_dataset = DynamicVideoDataset(args, scenes=args.eval_scenes)
  args.num_frames = test_dataset.num_frames

  # Create ibrnet model
  model = DynibarMono(args)
  eval_dataset_name = args.eval_dataset
  extra_out_dir = '{}/{}/{}'.format(
      eval_dataset_name, args.expname, str(args.render_idx)
  )
  print('saving results to {}...'.format(extra_out_dir))
  os.makedirs(extra_out_dir, exist_ok=True)

  projector = Projector(device='cuda:0')

  assert len(args.eval_scenes) == 1, 'only accept single scene'
  scene_name = args.eval_scenes[0]
  out_scene_dir = os.path.join(
      extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step), 'videos'
  )
  print('saving results to {}'.format(out_scene_dir))

  os.makedirs(out_scene_dir, exist_ok=True)
  os.makedirs(os.path.join(out_scene_dir, 'rgb_out'), exist_ok=True)

  save_prefix = scene_name
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  total_num = len(test_loader)
  out_frames = []
  full_frames = []
  crop_ratio = 0.03

  for i, data in enumerate(test_loader):
    idx = int(data['id'].item())
    start = time.time()
    ref_time_embedding = data['ref_time'].cuda()
    ref_frame_idx = int(data['id'].item())
    ref_time_offset = [
        int(near_idx - ref_frame_idx)
        for near_idx in data['nearest_pose_ids'].squeeze().tolist()
    ]

    model.switch_to_eval()
    with torch.no_grad():
      ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
      ray_batch = ray_sampler.get_all()

      cb_featmaps_1, cb_featmaps_2 = model.feature_net(
          ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
      )
      ref_featmaps = cb_featmaps_1  # [0:NUM_DYNAMIC_SRC_VIEWS]

      static_src_rgbs = (
          ray_batch['static_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
      )
      static_featmaps, _ = model.feature_net_st(static_src_rgbs)

      ret = render_single_image_mono(
          frame_idx=(ref_frame_idx, None),
          time_embedding=(ref_time_embedding, None),
          time_offset=(ref_time_offset, None),
          ray_sampler=ray_sampler,
          ray_batch=ray_batch,
          model=model,
          projector=projector,
          chunk_size=args.chunk_size,
          det=True,
          N_samples=args.N_samples,
          args=args,
          inv_uniform=args.inv_uniform,
          N_importance=args.N_importance,
          white_bkgd=args.white_bkgd,
          featmaps=(ref_featmaps, None, static_featmaps),
          is_train=False,
          num_vv=args.num_vv
      )

    coarse_pred_rgb = ret['outputs_coarse_ref']['rgb'].detach().cpu()
    coarse_pred_rgb_st = ret['outputs_coarse_ref']['rgb_static'].detach().cpu()
    coarse_pred_rgb_rgb = ret['outputs_coarse_ref']['rgb_dy'].detach().cpu()

    coarse_pred_rgb = (
        255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.0)
    ).astype(np.uint8)

    h, w = coarse_pred_rgb.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    coarse_pred_rgb = coarse_pred_rgb[crop_h:h-crop_h, crop_w:w-crop_w, ...]

    gt_rgb = data['rgb'][0, crop_h:h-crop_h, crop_w:w-crop_w, ...]
    gt_rgb = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)

    full_rgb = np.concatenate([gt_rgb, coarse_pred_rgb], axis=1)

    full_frames.append(coarse_pred_rgb)

    imageio.imwrite(os.path.join(out_scene_dir, 'rgb_out', '{}.png'.format(i)),
                    coarse_pred_rgb)

    print('frame {} completed, {}'.format(i, time.time() - start))
