"""Forward-Facing data loading code.

Modify from IBRNet
github.com/googleinterns/IBRNet/blob/master/ibrnet/data_loaders/llff_data_utils.py
"""

import os

import cv2
import imageio
import numpy as np


def parse_llff_pose(pose):
  """convert llff format pose to 4x4 matrix of intrinsics and extrinsics."""

  h, w, f = pose[:3, -1]
  c2w = pose[:3, :4]
  c2w_4x4 = np.eye(4)
  c2w_4x4[:3] = c2w
  c2w_4x4[:, 1:3] *= -1
  intrinsics = np.array(
      [[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  )
  return intrinsics, c2w_4x4


def batch_parse_llff_poses(poses):
  """Parse LLFF data format to opencv/colmap format."""
  all_intrinsics = []
  all_c2w_mats = []
  for pose in poses:
    intrinsics, c2w_mat = parse_llff_pose(pose)
    all_intrinsics.append(intrinsics)
    all_c2w_mats.append(c2w_mat)
  all_intrinsics = np.stack(all_intrinsics)
  all_c2w_mats = np.stack(all_c2w_mats)
  return all_intrinsics, all_c2w_mats


def batch_parse_vv_poses(poses):
  """Parse virtural views pose used for monocular video training."""
  all_c2w_mats = []
  for pose in poses:
    t_c2w_mats = []
    for p in pose:
      intrinsics, c2w_mat = parse_llff_pose(p)
      t_c2w_mats.append(c2w_mat)
    t_c2w_mats = np.stack(t_c2w_mats)
    all_c2w_mats.append(t_c2w_mats)

  all_c2w_mats = np.stack(all_c2w_mats)

  return all_c2w_mats


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
  """Function for loading LLFF data."""
  poses_arr = np.load(os.path.join(basedir, 'poses_bounds_cvd.npy'))
  poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
  bds = poses_arr[:, -2:].transpose([1, 0])

  img0 = [
      os.path.join(basedir, 'images', f)
      for f in sorted(os.listdir(os.path.join(basedir, 'images')))
      if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
  ][0]
  sh = imageio.imread(img0).shape

  sfx = ''

  if factor is not None and factor != 1:
    sfx = '_{}'.format(factor)
  elif height is not None:
    factor = sh[0] / float(height)
    width = int(round(sh[1] / factor))
    sfx = '_{}x{}'.format(width, height)
  elif width is not None:
    factor = sh[1] / float(width)
    height = int(round(sh[0] / factor))
    sfx = '_{}x{}'.format(width, height)
  else:
    factor = 1

  imgdir = os.path.join(basedir, 'images' + sfx)
  print('imgdir ', imgdir, ' factor ', factor)

  if not os.path.exists(imgdir):
    print(imgdir, 'does not exist, returning')
    return

  imgfiles = [
      os.path.join(imgdir, f)
      for f in sorted(os.listdir(imgdir))
      if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
  ]

  if poses.shape[-1] != len(imgfiles):
    print(
        '{}: Mismatch between imgs {} and poses {} !!!!'.format(
            basedir, len(imgfiles), poses.shape[-1]
        )
    )
    raise NotImplementedError

  sh = imageio.imread(imgfiles[0]).shape
  poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
  poses[2, 4, :] = poses[2, 4, :]  # * 1. / factor

  def imread(f):
    if f.endswith('png'):
      return imageio.imread(f, ignoregamma=True)
    else:
      return imageio.imread(f)

  if not load_imgs:
    imgs = None
  else:
    imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
    imgs = np.stack(imgs, -1)
    print('Loaded image data', imgs.shape, poses[:, -1, 0])

  return poses, bds, imgs, imgfiles


def normalize(x):
  return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
  vec2 = normalize(z)
  vec1_avg = up
  vec0 = normalize(np.cross(vec1_avg, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, pos], 1)
  return m


def ptstocam(pts, c2w):
  tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
  return tt


def poses_avg(poses):
  hwf = poses[0, :3, -1:]

  center = poses[:, :3, 3].mean(0)
  vec2 = normalize(poses[:, :3, 2].sum(0))
  up = poses[:, :3, 1].sum(0)
  c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

  return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
  """Render a spiral path."""

  render_poses = []
  rads = np.array(list(rads) + [1.0])
  hwf = c2w[:, 4:5]

  for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
    c = np.dot(
        c2w[:3, :4],
        np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
        * rads,
    )
    z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
    render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
  return render_poses


def recenter_poses(poses):
  """Recenter camera poses into centroid."""
  poses_ = poses + 0
  bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
  c2w = poses_avg(poses)
  c2w = np.concatenate([c2w[:3, :4], bottom], -2)
  bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
  poses = np.concatenate([poses[:, :3, :4], bottom], -2)

  poses = np.linalg.inv(c2w) @ poses
  poses_[:, :3, :4] = poses[:, :3, :4]
  poses = poses_
  return poses


def recenter_poses_mono(poses, src_vv_poses):
  """Recenter virutal view camera poses into centroid."""
  hwf = poses[:, :, 4:5]
  poses_ = poses + 0
  bottom = np.reshape([0, 0, 0, 1.], [1, 4])
  c2w = poses_avg(poses)
  c2w = np.concatenate([c2w[:3, :4], bottom], -2)
  bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
  poses = np.concatenate([poses[:, :3, :4], bottom], -2)

  poses = np.linalg.inv(c2w) @ poses
  poses_[:, :3, :4] = poses[:, :3, :4]
  poses = poses_

  src_output_poses = np.zeros((
      src_vv_poses.shape[1],
      src_vv_poses.shape[0],
      src_vv_poses.shape[2],
      src_vv_poses.shape[3] + 1,
  ))
  for i in range(src_vv_poses.shape[1]):
    src_vv_poses_ = np.concatenate([src_vv_poses[:, i, :3, :4], bottom], -2)
    src_vv_poses_ = np.linalg.inv(c2w) @ src_vv_poses_
    src_output_poses[i, ...] = np.concatenate([src_vv_poses_[:, :3, :], hwf], 2)

  return poses, np.moveaxis(src_output_poses, 1, 0)


def load_llff_data(
    basedir,
    height,
    num_avg_imgs,
    factor=8,
    render_idx=8,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    load_imgs=True,
):
  """Load LLFF forward-facing data.
  
  Args:
    basedir: base directory
    height: training image height
    factor: resize factor
    render_idx: rendering frame index from the video
    recenter: recentor camera poses
    bd_factor: scale factor for bounds
    spherify: spherify the camera poses
    load_imgs: load images from the disk

  Returns:
    images: video frames
    poses: corresponding camera parameters
    bds: bounds
    render_poses: rendering camera poses 
    i_test: test index
    imgfiles: list of image path
    scale: scene scale
  """
  out = _load_data(
      basedir, factor=None, load_imgs=load_imgs, height=height
  )

  if out is None:
    return
  else:
    poses, bds, imgs, imgfiles = out

  # Correct rotation matrix ordering and move variable dim to axis 0
  poses = np.concatenate(
      [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
  )
  poses = np.moveaxis(poses, -1, 0).astype(np.float32)
  if imgs is not None:
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    images = images.astype(np.float32)
  else:
    images = None

  bds = np.moveaxis(bds, -1, 0).astype(np.float32)

  # Rescale if bd_factor is provided
  scale = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)

  poses[:, :3, 3] *= scale
  bds *= scale

  if recenter:
    poses = recenter_poses(poses)

  spiral = True
  if spiral:
    print('================= render_path_spiral ==========================')
    c2w = poses_avg(poses[0:num_avg_imgs])
    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * 0.9, bds.max() * 2.0
    dt = 0.75
    mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    focal = mean_dz * 1.5

    # Get radii for spiral path
    # shrink_factor = 0.8
    zdelta = close_depth * 0.2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 80, 0)
    c2w_path = c2w
    n_views = 120
    n_rots = 2

    # Generate poses for spiral path
    render_poses = render_path_spiral(
        c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=n_rots, N=n_views
    )
  else:
    raise NotImplementedError

  render_poses = np.array(render_poses).astype(np.float32)

  dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
  i_test = np.argmin(dists)
  poses = poses.astype(np.float32)

  print('bds ', bds.min(), bds.max())

  return images, poses, bds, render_poses, i_test, imgfiles, scale


def load_mono_data(
    basedir,
    height=288,
    factor=8,
    render_idx=-1,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    load_imgs=True,
):
  """Load monocular video data.
  
  Args:
    basedir: base directory
    height: training image height
    factor: resize factor
    render_idx: rendering frame index from the video
    recenter: recentor camera poses
    bd_factor: scale factor for bounds
    spherify: spherify the camera poses
    load_imgs: load images from the disk

  Returns:
    images: video frames
    poses: corresponding camera parameters
    src_vv_poses: virtual view camera poses
    bds: bounds
    render_poses: rendering camera poses 
    i_test: test index
    imgfiles: list of image path
    scale: scene scale
  """
  out = _load_data(basedir, factor=None, load_imgs=load_imgs, height=height)

  src_vv_poses = np.load(os.path.join(basedir, 'source_vv_poses.npy'))

  if out is None:
    return
  else:
    poses, bds, imgs, imgfiles = out

  # Correct rotation matrix ordering and move variable dim to axis 0
  poses = np.concatenate(
      [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
  )
  src_vv_poses = np.concatenate(
      [
          src_vv_poses[:, :, 1:2, :],
          -src_vv_poses[:, :, 0:1, :],
          src_vv_poses[:, :, 2:, :],
      ],
      2,
  )

  poses = np.moveaxis(poses, -1, 0).astype(np.float32)
  src_vv_poses = np.moveaxis(src_vv_poses, -1, 0).astype(np.float32)

  if imgs is not None:
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    images = images.astype(np.float32)
  else:
    images = None

  bds = np.moveaxis(bds, -1, 0).astype(np.float32)

  # Rescale if bd_factor is provided
  scale = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)

  poses[:, :3, 3] *= scale
  src_vv_poses[..., :3, 3] *= scale

  bds *= scale

  if recenter:
    poses, src_vv_poses = recenter_poses_mono(poses, src_vv_poses)

  if render_idx >= 0:
    render_poses = render_wander_path(poses[render_idx])
  else:
    render_poses = render_stabilization_path(poses, k_size=45)

  render_poses = np.array(render_poses).astype(np.float32)

  i_test = []
  poses = poses.astype(np.float32)

  print('bds ', bds.min(), bds.max())

  return images, poses, src_vv_poses, bds, render_poses, i_test, imgfiles, scale


def render_wander_path(c2w):
  """Rendering circular path."""
  hwf = c2w[:, 4:5]
  num_frames = 50
  max_disp = 48.0

  max_trans = max_disp / hwf[2][0]
  output_poses = []

  for i in range(num_frames):
    x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
    y_trans = 0.#max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 2.
    z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 2.

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

    i_pose = np.linalg.inv(i_pose)

    ref_pose = np.concatenate(
        [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
    )

    render_pose = np.dot(ref_pose, i_pose)
    output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))

  return output_poses


def render_stabilization_path(poses, k_size):
  """Rendering stablizaed camera path."""

  hwf = poses[0, :, 4:5]
  num_frames = poses.shape[0]
  output_poses = []

  input_poses = []

  for i in range(num_frames):
    input_poses.append(
        np.concatenate(
            [poses[i, :3, 0:1], poses[i, :3, 1:2], poses[i, :3, 3:4]], axis=-1
        )
    )

  input_poses = np.array(input_poses)

  gaussian_kernel = cv2.getGaussianKernel(
      ksize=k_size, sigma=-1
  )
  output_r1 = cv2.filter2D(input_poses[:, :, 0], -1, gaussian_kernel)
  output_r2 = cv2.filter2D(input_poses[:, :, 1], -1, gaussian_kernel)

  output_r1 = output_r1 / np.linalg.norm(output_r1, axis=-1, keepdims=True)
  output_r2 = output_r2 / np.linalg.norm(output_r2, axis=-1, keepdims=True)

  output_t = cv2.filter2D(input_poses[:, :, 2], -1, gaussian_kernel)

  for i in range(num_frames):
    output_r3 = np.cross(output_r1[i], output_r2[i])

    render_pose = np.concatenate(
        [
            output_r1[i, :, None],
            output_r2[i, :, None],
            output_r3[:, None],
            output_t[i, :, None],
        ],
        axis=-1,
    )

    output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))

  return output_poses
