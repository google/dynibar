"""Dataloader class for training monocular videos."""


import os
import cv2
from ibrnet.data_loaders.data_utils import get_nearest_pose_ids
from ibrnet.data_loaders.llff_data_utils import batch_parse_llff_poses
from ibrnet.data_loaders.llff_data_utils import batch_parse_vv_poses
from ibrnet.data_loaders.llff_data_utils import load_mono_data
import imageio
import numpy as np
import skimage.morphology
import torch
from torch.utils.data import Dataset


class MonocularDataset(Dataset):
  """This class loads data from monocular video.
  
  Each returned item in the dataset has 
    id: reference frame index
    anchor_id: nearby frame index for cross time rendering
    num_frames: number of video frames
    ref_time: normalized reference time index
    anchor_time: normalized nearby cross-time index
    nearest_pose_ids: source view index w.r.t reference time
    anchor_nearest_pose_ids: source view index w.r.t nearby time
    rgb: [H, W, 3], image at reference time
    disp: [H, W], disparity at reference time
    motion_mask: [H, W], dynamic mask at reference time
    static_mask: [H, W], static mask at reference time
    flows: [6, H, W, 2] optical flows from reference time
    masks: [6, H, W] optical flow valid masks from reference time
    camera: [34] camera parameters at reference time
    anchor_camera: [34] camera parameters at nearby cross-time
    rgb_path: RGB file path name
    src_rgbs: [..., H, W, 3] source views RGB images for dynamic model
    src_cameras: [..., 34] source view camera parameters for dynamic model
    static_src_rgbs: [..., H, W, 3] srouce view images for static model
    static_src_cameras: [..., 34] source view camera parameters for static model
    anchor_src_rgbs: [..., H, W, 3] cross-time view images for dynamic model
    anchor_src_cameras: [..., 34] cross-time source view camera parameters for
    dynamic model
    depth_range: [2] scene near and far bounds
  """

  def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
    assert len(scenes) == 1
    self.folder_path = args.folder_path
    self.num_vv = args.num_vv
    self.args = args
    self.mask_src_view = args.mask_src_view
    self.num_frames_sample = args.num_source_views
    self.erosion_radius = args.erosion_radius
    self.random_crop = random_crop

    self.max_range = args.max_range

    scene = scenes[0]
    self.scene_path = os.path.join(self.folder_path, scene, 'dense')
    _, poses, src_vv_poses, bds, _, _, rgb_files, scale = (
        load_mono_data(
            self.scene_path, height=args.training_height, load_imgs=False
        )
    )
    near_depth = np.min(bds)

    # make sure far scenes to be at least 15
    # so that static model is able to model view-dependent effect.
    if np.max(bds) < 10:
      far_depth = min(20, np.max(bds) + 15.0)
    else:
      far_depth = min(50, max(20, np.max(bds)))

    print('============= FINAL NEAR FAR', near_depth, far_depth)

    intrinsics, c2w_mats = batch_parse_llff_poses(poses)
    self.src_vv_c2w_mats = batch_parse_vv_poses(src_vv_poses)
    self.num_frames = len(rgb_files)
    assert self.num_frames == poses.shape[0]
    i_train = np.arange(self.num_frames)
    i_render = i_train
    self.scale = scale

    num_render = len(i_render)
    self.train_rgb_files = rgb_files
    self.train_intrinsics = intrinsics
    self.train_poses = c2w_mats
    self.train_depth_range = [[near_depth, far_depth]] * num_render

  def read_optical_flow(self, basedir, img_i, start_frame, fwd, interval):
    flow_dir = os.path.join(basedir, 'flow_i%d' % interval)

    if fwd:
      fwd_flow_path = os.path.join(
          flow_dir, '%05d_fwd.npz' % (start_frame + img_i)
      )
      fwd_data = np.load(fwd_flow_path)  # , (w, h))
      fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
      fwd_mask = np.float32(fwd_mask)

      return fwd_flow, fwd_mask
    else:
      bwd_flow_path = os.path.join(
          flow_dir, '%05d_bwd.npz' % (start_frame + img_i)
      )

      bwd_data = np.load(bwd_flow_path)  # , (w, h))
      bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
      bwd_mask = np.float32(bwd_mask)

      return bwd_flow, bwd_mask

  def __len__(self):
    return self.num_frames

  def set_epoch(self, epoch):
    self.current_epoch = epoch

  def load_src_view(
      self, rgb_file, pose, intrinsics, st_mask_path=None
  ):
    """Load RGB and camera data from each source views id."""

    src_rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
    img_size = src_rgb.shape[:2]
    src_camera = np.concatenate(
        (list(img_size), intrinsics.flatten(), pose.flatten())
    ).astype(np.float32)

    if st_mask_path:
      st_mask = imageio.imread(st_mask_path).astype(np.float32) / 255.0
      st_mask = cv2.resize(
          st_mask,
          (src_rgb.shape[1], src_rgb.shape[0]),
          interpolation=cv2.INTER_NEAREST,
      )

      if len(st_mask.shape) == 2:
        st_mask = st_mask[..., None]

      src_rgb = src_rgb * st_mask

    return src_rgb, src_camera

  def __getitem__(self, idx):
    # skip first and last 3 frames
    idx = np.random.randint(3, self.num_frames - 3)
    rgb_file = self.train_rgb_files[idx]

    render_pose = self.train_poses[idx]
    intrinsics = self.train_intrinsics[idx]
    depth_range = self.train_depth_range[idx]

    rgb, camera = self.load_src_view(rgb_file, render_pose, intrinsics)
    img_size = rgb.shape[:2]

    # load mono-depth
    disp_path = os.path.join(
        self.scene_path, 'disp', rgb_file.split('/')[-1][:-4] + '.npy'
    )
    disp = np.load(disp_path) / self.scale

    # load motion mask
    mask_path = os.path.join(
        '/'.join(rgb_file.split('/')[:-2]), 'dynamic_masks', '%d.png' % idx
    )
    motion_mask = 1.0 - imageio.imread(mask_path).astype(np.float32) / 255.0

    static_mask_path = os.path.join(
        '/'.join(rgb_file.split('/')[:-2]), 'static_masks', '%d.png' % idx
    )
    static_mask = (
        1.0 - imageio.imread(static_mask_path).astype(np.float32) / 255.0
    )

    static_mask = cv2.resize(
        static_mask,
        (disp.shape[1], disp.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    # ensure input dynamic and static mask to have same height before
    # running morphological erosion
    motion_mask = cv2.resize(
        motion_mask,
        (int(round(288.0 * disp.shape[1] / disp.shape[0])), 288),
        interpolation=cv2.INTER_NEAREST,
    )

    if len(motion_mask.shape) == 2:
      motion_mask = motion_mask[..., None]

    motion_mask = skimage.morphology.erosion(
        motion_mask[..., 0] > 1e-3, skimage.morphology.disk(self.erosion_radius)
    )

    motion_mask = cv2.resize(
        np.float32(motion_mask),
        (disp.shape[1], disp.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    motion_mask = np.float32(motion_mask)
    static_mask = np.float32(static_mask > 1e-3)

    assert disp.shape[0:2] == img_size
    assert motion_mask.shape[0:2] == img_size
    assert static_mask.shape[0:2] == img_size

    # train_set_id = self.render_train_set_ids[idx]
    train_rgb_files = self.train_rgb_files
    train_poses = self.train_poses
    train_intrinsics = self.train_intrinsics

    # view selection based on time interval
    nearest_pose_ids = [idx + offset for offset in [1, 2, 3, -1, -2, -3]]
    max_step = min(3, self.current_epoch // (self.args.init_decay_epoch) + 1)
    # select a nearby time index for cross time rendering
    anchor_pool = [i for i in range(1, max_step + 1)] + [
        -i for i in range(1, max_step + 1)
    ]
    anchor_idx = idx + anchor_pool[np.random.choice(len(anchor_pool))]
    anchor_nearest_pose_ids = []

    anchor_camera = np.concatenate((
        list(img_size),
        self.train_intrinsics[anchor_idx].flatten(),
        self.train_poses[anchor_idx].flatten(),
    )).astype(np.float32)

    for offset in [3, 2, 1, 0, -1, -2, -3]:
      if (
          (anchor_idx + offset) < 0
          or (anchor_idx + offset) >= len(train_rgb_files)
          or (anchor_idx + offset) == idx
      ):
        continue
      anchor_nearest_pose_ids.append((anchor_idx + offset))

    # occasionally include render image for anchor time index
    if np.random.choice([0, 1], p=[1.0 - 0.005, 0.005]):
      anchor_nearest_pose_ids.append(idx)

    anchor_nearest_pose_ids = np.sort(anchor_nearest_pose_ids)

    flows, masks = [], []

    # load optical flow
    for ii in range(len(nearest_pose_ids)):
      offset = nearest_pose_ids[ii] - idx
      flow, mask = self.read_optical_flow(
          self.scene_path,
          idx,
          start_frame=0,
          fwd=True if offset > 0 else False,
          interval=np.abs(offset),
      )

      flows.append(flow)
      masks.append(mask)

    flows = np.stack(flows)
    masks = np.stack(masks)

    assert flows.shape[1:3] == img_size
    assert masks.shape[1:3] == img_size

    # load src rgb for ref view
    sp_pose_ids = get_nearest_pose_ids(
        render_pose,
        train_poses,
        tar_id=idx,
        angular_dist_method='dist',
    )

    static_pose_ids = []

    max_interval = self.max_range // self.num_frames_sample
    interval = np.random.randint(max(2, max_interval - 2), max_interval + 1)

    for ii in range(-self.num_frames_sample, self.num_frames_sample):
      rand_j = np.random.randint(1, interval + 1)
      static_pose_id = idx + interval * ii + rand_j

      if 0 <= static_pose_id < self.num_frames and static_pose_id != idx:
        static_pose_ids.append(static_pose_id)

    static_pose_set = set(static_pose_ids)
    # if there are no enough image, add nearest images w.r.t camera poses
    # choose stride of 5 so that views are not very close to each other.
    for sp_pose_id in sp_pose_ids[::5]:
      if len(static_pose_ids) >= (self.num_frames_sample * 2):
        break

      if sp_pose_id not in static_pose_set:
        static_pose_ids.append(sp_pose_id)

    static_pose_ids = np.sort(static_pose_ids)

    src_rgbs = []
    src_cameras = []

    for near_id in nearest_pose_ids:
      src_rgb, src_camera = self.load_src_view(
          train_rgb_files[near_id],
          train_poses[near_id],
          train_intrinsics[near_id],
      )
      src_rgbs.append(src_rgb)
      src_cameras.append(src_camera)

    # load src virtual views
    for virtual_idx in np.random.choice(
        list(range(0, 8)), size=self.num_vv, replace=False
    ):
      src_vv_path = os.path.join(
          '/'.join(
              rgb_file.replace('images', 'source_virtual_views').split('/')[:-1]
          ),
          '%05d' % idx,
          '%02d.png' % virtual_idx,
      )
      src_rgb, src_camera = self.load_src_view(
          src_vv_path,
          self.src_vv_c2w_mats[idx, virtual_idx],
          intrinsics,
      )
      src_rgbs.append(src_rgb)
      src_cameras.append(src_camera)

    src_rgbs = np.stack(src_rgbs, axis=0)
    src_cameras = np.stack(src_cameras, axis=0)

    static_src_rgbs = []
    static_src_cameras = []

    # load src rgb for static view
    for st_near_id in static_pose_ids:
      st_mask_path = None

      if self.mask_src_view:
        st_mask_path = os.path.join(
            '/'.join(rgb_file.split('/')[:-2]),
            'dynamic_masks',
            '%d.png' % st_near_id,
        )

      src_rgb, src_camera = self.load_src_view(
          train_rgb_files[st_near_id],
          train_poses[st_near_id],
          train_intrinsics[st_near_id],
          st_mask_path=st_mask_path,
      )

      static_src_rgbs.append(src_rgb)
      static_src_cameras.append(src_camera)

    static_src_rgbs = np.stack(static_src_rgbs, axis=0)
    static_src_cameras = np.stack(static_src_cameras, axis=0)

    # load src rgb for anchor view
    anchor_src_rgbs = []
    anchor_src_cameras = []

    for near_id in anchor_nearest_pose_ids:
      src_rgb, src_camera = self.load_src_view(
          train_rgb_files[near_id],
          train_poses[near_id],
          train_intrinsics[near_id],
      )
      anchor_src_rgbs.append(src_rgb)
      anchor_src_cameras.append(src_camera)

    # load anchor src virtual views
    for virtual_idx in np.random.choice(
        list(range(0, 8)), size=self.num_vv, replace=False
    ):
      src_vv_path = os.path.join(
          '/'.join(
              rgb_file.replace('images', 'source_virtual_views').split('/')[:-1]
          ),
          '%05d' % anchor_idx,
          '%02d.png' % virtual_idx,
      )
      src_rgb, src_camera = self.load_src_view(
          src_vv_path,
          self.src_vv_c2w_mats[anchor_idx, virtual_idx],
          intrinsics,
      )
      anchor_src_rgbs.append(src_rgb)
      anchor_src_cameras.append(src_camera)

    anchor_src_rgbs = np.stack(anchor_src_rgbs, axis=0)
    anchor_src_cameras = np.stack(anchor_src_cameras, axis=0)

    depth_range = torch.tensor(
        [depth_range[0] * 0.9, depth_range[1] * 1.5]
    ).float()

    return {
        'id': idx,
        'anchor_id': anchor_idx,
        'num_frames': self.num_frames,
        'ref_time': float(idx / float(self.num_frames)),
        'anchor_time': float(anchor_idx / float(self.num_frames)),
        'nearest_pose_ids': torch.from_numpy(np.array(nearest_pose_ids)),
        'anchor_nearest_pose_ids': torch.from_numpy(
            np.array(anchor_nearest_pose_ids)
        ),
        'rgb': torch.from_numpy(rgb[..., 0:3]).float(),
        'disp': torch.from_numpy(disp).float(),
        'motion_mask': torch.from_numpy(motion_mask).float(),
        'static_mask': torch.from_numpy(static_mask).float(),
        'flows': torch.from_numpy(flows).float(),
        'masks': torch.from_numpy(masks).float(),
        'camera': torch.from_numpy(camera).float(),
        'anchor_camera': torch.from_numpy(anchor_camera).float(),
        'rgb_path': rgb_file,
        'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
        'src_cameras': torch.from_numpy(src_cameras).float(),
        'static_src_rgbs': torch.from_numpy(static_src_rgbs[..., :3]).float(),
        'static_src_cameras': torch.from_numpy(static_src_cameras).float(),
        'anchor_src_rgbs': torch.from_numpy(anchor_src_rgbs[..., :3]).float(),
        'anchor_src_cameras': torch.from_numpy(anchor_src_cameras).float(),
        'depth_range': depth_range,
    }
