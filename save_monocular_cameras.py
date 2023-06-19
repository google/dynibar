"""Save images, depth, flow and mask data into dynibar input format."""

'''
<!-- Preprocessing:

python render_source_vv.py \
--data_dir /home/zhengqili/filestore/NSFF/nerf_data/release/kid-running \
--cvd_dir
/home/zhengqili/filestore/dynamic-video-DPT/monocular-results/kid-runningscene_flow_motion_field_shutterstock_epoch_15/epoch0015_test

python save_monocular_cameras.py \
--data_dir /home/zhengqili/filestore/NSFF/nerf_data/release/kid-running  \
--cvd_dir
/home/zhengqili/filestore/dynamic-video-DPT/monocular-results/kid-runningscene_flow_motion_field_shutterstock_epoch_15/epoch0015_test -->


'''


import argparse
import glob
import os
import cv2
import imageio
import numpy as np


SAVE_IMG = True
FINAL_H = 288

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cvd_dir', type=str, help='depth directory')
  parser.add_argument('--data_dir', type=str, help='dataset directory') 
  # parser.add_argument("--scene_name", type=str, 
                      # help='Scene name') # 'kid-running'
  args = parser.parse_args()

  pt_out_list = sorted(glob.glob(os.path.join(args.cvd_dir, '*.npz')))
  data_dir = os.path.join(args.data_dir, 'dense')

  try:
    original_img_path = os.path.join(data_dir, 'images', '00000.png')
    o_img = imageio.imread(original_img_path)
  except:
    original_img_path = os.path.join(data_dir, 'images', '00000.jpg')
    o_img = imageio.imread(original_img_path)

  o_ar = float(o_img.shape[1]) / float(o_img.shape[0])

  final_w, final_h = int(round(FINAL_H * o_ar)), int(FINAL_H)

  img_dir = os.path.join(data_dir, 'images_%dx%d' % (final_w, final_h))
  os.makedirs(img_dir, exist_ok=True)
  print('img_dir ', img_dir)
  disp_dir = os.path.join(data_dir, 'disp')
  os.makedirs(disp_dir, exist_ok=True)

  Ks = []
  mono_depths = []
  c2w_mats = []
  imgs = []
  bounds_mats = []

  for i, pt_out_path in enumerate(pt_out_list):
    print(i)
    out_name = pt_out_path.split('/')[-1]
    pt_data = np.load(pt_out_path)

    img = pt_data['img_1'][0].transpose(1, 2, 0)
    pred_depth = pt_data['depth'][0, 0, ...]
    pred_disp = 1.0 / pred_depth
    K = pt_data['K'][0, 0, 0, ...].transpose()
    img = pt_data['img_1'][0].transpose(1, 2, 0)
    cam_c2w = pt_data['cam_c2w'][0]

    K[0, :] *= final_w / img.shape[1]
    K[1, :] *= final_h / img.shape[0]

    print('K ', K, abs(K[0, 0] - K[1, 1]) / (K[1, 1] + K[0, 0]))
    assert (
        abs(K[0, 0] - K[1, 1]) / (K[1, 1] + K[0, 0]) < 0.005
    )  # we assume fx ~= fy

    original_img_path = os.path.join(
        data_dir, 'images', '%05d.png' % int(out_name[5:9])
    )
    o_img = imageio.imread(original_img_path)
    print(o_img.shape, final_w, final_h)
    img_resized = cv2.resize(
        o_img, (final_w, final_h), interpolation=cv2.INTER_AREA
    )
    pred_disp_resized = cv2.resize(
        pred_disp, (final_w, final_h), interpolation=cv2.INTER_LINEAR
    )

    if SAVE_IMG:
      imageio.imwrite(os.path.join(img_dir, '%05d.png' % i), img_resized)
      np.save(
          os.path.join(disp_dir, '%05d.npy' % i),
          pred_disp_resized.astype(np.float32),
      )

    mono_depths.append(pred_depth)
    c2w_mats.append(cam_c2w)
    imgs.append(img_resized)

    close_depth, inf_depth = np.percentile(pred_depth, 5), np.percentile(
        pred_depth, 95
    )
    # print(close_depth, inf_depth)
    bounds = np.array([close_depth, inf_depth])
    bounds_mats.append(bounds)

  c2w_mats = np.stack(c2w_mats, 0)
  bounds_mats = np.stack(bounds_mats, 0)

  h, w, fx, fy = imgs[0].shape[0], imgs[0].shape[1], K[0, 0], K[1, 1]

  print('h, w ', h, w, fx, fy)
  print('bounds_mats ', np.min(bounds_mats), np.max(bounds_mats))

  ff = (fx + fy) / 2.0
  # hwf = np.array([h, w, fx, fy]).reshape([1, 4])
  hwf = np.array([h, w, ff]).reshape([3, 1])

  poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])

  poses = np.concatenate(
      [poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1
  )

  # must switch to [-y, x, z] from [x, -y, -z], NOT [r, u, -t]
  poses = np.concatenate(
      [
          poses[:, 1:2, :],
          poses[:, 0:1, :],
          -poses[:, 2:3, :],
          poses[:, 3:4, :],
          poses[:, 4:5, :],
      ],
      1,
  )

  save_arr = []
  for i in range((poses.shape[2])):
    save_arr.append(np.concatenate([poses[..., i].ravel(), bounds_mats[i]], 0))

  np.save(os.path.join(data_dir, 'poses_bounds_cvd.npy'), save_arr)
