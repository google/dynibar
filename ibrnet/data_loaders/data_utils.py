"""utility function definition for data loader."""

import math
import numpy as np

rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def vector_norm(data, axis=None, out=None):
  """Return length, i.e. eucledian norm, of ndarray along axis."""
  data = np.array(data, dtype=np.float64, copy=True)
  if out is None:
    if data.ndim == 1:
      return math.sqrt(np.dot(data, data))
    data *= data
    out = np.atleast_1d(np.sum(data, axis=axis))
    np.sqrt(out, out)
    return out
  else:
    data *= data
    np.sum(data, axis=axis, out=out)
    np.sqrt(out, out)


def quaternion_about_axis(angle, axis):
  """Return quaternion for rotation about axis."""
  quaternion = np.zeros((4,), dtype=np.float64)
  quaternion[:3] = axis[:3]
  qlen = vector_norm(quaternion)
  if qlen > _EPS:
    quaternion *= math.sin(angle / 2.0) / qlen
  quaternion[3] = math.cos(angle / 2.0)
  return quaternion


def quaternion_matrix(quaternion):
  """Return homogeneous rotation matrix from quaternion."""
  q = np.array(quaternion[:4], dtype=np.float64, copy=True)
  nq = np.dot(q, q)
  if nq < _EPS:
    return np.identity(4)
  q *= math.sqrt(2.0 / nq)
  q = np.outer(q, q)
  return np.array(
      (
          (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
          (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
          (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
          (0.0, 0.0, 0.0, 1.0),
      ),
      dtype=np.float64,
  )


def angular_dist_between_2_vectors(vec1, vec2):
  vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
  vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
  angular_dists = np.arccos(
      np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0)
  )
  return angular_dists


def batched_angular_dist_rot_matrix(r1, r2):
  """calculate the angular distance between two rotation matrices (batched)."""

  assert (
      r1.shape[-1] == 3
      and r2.shape[-1] == 3
      and r1.shape[-2] == 3
      and r2.shape[-2] == 3
  )
  return np.arccos(
      np.clip(
          (np.trace(np.matmul(r2.transpose(0, 2, 1), r1), axis1=1, axis2=2) - 1)
          / 2.0,
          a_min=-1 + TINY_NUMBER,
          a_max=1 - TINY_NUMBER,
      )
  )


def get_nearest_pose_ids(
    tar_pose,
    ref_poses,
    tar_id=-1,
    angular_dist_method='vector',
    scene_center=(0, 0, 0),
):
  """Get poses id in nearest neighboorhood manner."""
  num_cams = len(ref_poses)
  batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

  if angular_dist_method == 'matrix':
    dists = batched_angular_dist_rot_matrix(
        batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3]
    )
  elif angular_dist_method == 'vector':
    tar_cam_locs = batched_tar_pose[:, :3, 3]
    ref_cam_locs = ref_poses[:, :3, 3]
    scene_center = np.array(scene_center)[None, ...]
    tar_vectors = tar_cam_locs - scene_center
    ref_vectors = ref_cam_locs - scene_center
    dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
  elif angular_dist_method == 'dist':
    tar_cam_locs = batched_tar_pose[:, :3, 3]
    ref_cam_locs = ref_poses[:, :3, 3]
    dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
  else:
    raise NotImplementedError

  if tar_id >= 0:
    assert tar_id < num_cams
    dists[tar_id] = 1e3 

  sorted_ids = np.argsort(dists)

  return sorted_ids


def get_interval_pose_ids(
    tar_pose,
    ref_poses,
    tar_id=-1,
    angular_dist_method='dist',
    interval=2,
    scene_center=(0, 0, 0)):
  """Get poses id in nearest neighboorhood manner from every 'interval' frames."""

  original_indices = np.array(range(0, len(ref_poses)))

  ref_poses = ref_poses[::interval]
  subsample_indices = original_indices[::interval]

  num_cams = len(ref_poses)
  batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

  if angular_dist_method == 'matrix':
    dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], 
                                            ref_poses[:, :3, :3])
  elif angular_dist_method == 'vector':
    tar_cam_locs = batched_tar_pose[:, :3, 3]
    ref_cam_locs = ref_poses[:, :3, 3]
    scene_center = np.array(scene_center)[None, ...]
    tar_vectors = tar_cam_locs - scene_center
    ref_vectors = ref_cam_locs - scene_center
    dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
  elif angular_dist_method == 'dist':
    tar_cam_locs = batched_tar_pose[:, :3, 3]
    ref_cam_locs = ref_poses[:, :3, 3]
    dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
  else:
    raise NotImplementedError

  if tar_id >= 0:
    assert tar_id < num_cams
    dists[tar_id] = 1e3

  sorted_ids = np.argsort(dists)

  final_ids = subsample_indices[sorted_ids]

  return final_ids
