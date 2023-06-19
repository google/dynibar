"""Training script for monocular video."""

import os
import shutil
import time
import config
from ibrnet.criterion import Criterion
from ibrnet.criterion import compute_rgb_loss
from ibrnet.criterion import compute_temporal_rgb_loss
from ibrnet.criterion import compute_flow_loss

from ibrnet.data_loaders.create_training_dataset import create_training_dataset
from ibrnet.data_loaders.flow_utils import flow_to_image
from ibrnet.model import DynibarMono
from ibrnet.projection import Projector
from ibrnet.render_image import render_single_image_mono
from ibrnet.render_ray import render_rays_mono
from ibrnet.sample_ray import RaySamplerSingleImage
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_efficient_distloss import eff_distloss_native
from utils import colorize
from utils import img2mse
from utils import img_HWC2CHW
from utils import mse2psnr


def worker_init_fn(worker_id):
  np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
  """Helper function to synchronize (barrier) among all processes in distributed training."""
  if not dist.is_available():
    return
  if not dist.is_initialized():
    return
  world_size = dist.get_world_size()
  if world_size == 1:
    return
  dist.barrier()


def train(args):
  """Main train function."""
  torch.cuda.set_device(args.local_rank)
  args.expname = (
      args.expname
      + '_mr-%d' % (args.max_range)
      + '_w-disp-%.3f' % (args.w_disp)
      + '_w-flow-%.3f' % (args.w_flow)
      + '_anneal_cycle-%.1f-%.1f' % (args.w_cycle, args.cycle_factor)
      + '-w_mode-%d' % (args.occ_weights_mode)
  )

  device = 'cuda:{}'.format(args.local_rank)
  out_folder = os.path.join(args.rootdir, 'out', args.expname)
  print('outputs will be saved to {}'.format(out_folder))
  os.makedirs(out_folder, exist_ok=True)

  # save the args and config files
  f = os.path.join(out_folder, 'args.txt')
  with open(f, 'w') as file:
    for arg in sorted(vars(args)):
      attr = getattr(args, arg)
      file.write('{} = {}\n'.format(arg, attr))

  if args.config is not None:
    f = os.path.join(out_folder, 'config.txt')
    if not os.path.isfile(f):
      shutil.copy(args.config, f)

  # create training dataset
  train_dataset, train_sampler = create_training_dataset(args)
  # currently only support batch_size=1 
  # (i.e., one set of target and source views) for each GPU node
  # please use distributed parallel on multiple GPUs to train multiple 
  # target views per batch
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=1,
      worker_init_fn=lambda _: np.random.seed(),
      num_workers=args.workers,
      pin_memory=True,
      sampler=train_sampler,
      shuffle=True if train_sampler is None else False,
  )
  num_frames = args.num_frames = train_dataset.num_frames
  args.lrate_decay_steps = args.num_frames * args.init_decay_epoch

  # Create IBRNet model
  model = DynibarMono(
      args,
      # load_opt=not args.no_load_opt,
      # load_scheduler=not args.no_load_scheduler,
  )
  # create projector
  projector = Projector(device=device)

  # Create criterion
  rgb_criterion = Criterion()
  tb_dir = os.path.join(args.rootdir, 'logs/', args.expname)
  if args.local_rank == 0:
    writer = SummaryWriter(tb_dir)
    print('saving tensorboard files to {}'.format(tb_dir))
  scalars_to_log = {}

  global_step = model.start_step
  start_epoch = global_step // num_frames
  decay_rate = args.decay_rate

  # First bootstrap static model for better training stability 
  for epoch in range(start_epoch, args.init_decay_epoch // 2):
    train_dataset.set_epoch(epoch)
    print('================ Static Boostrap ', epoch)

    for ii, train_data in enumerate(train_loader):
      ref_time_embedding = train_data['ref_time'].to(device)
      anchor_time_embedding = train_data['anchor_time'].to(device)

      nearest_pose_ids = train_data['nearest_pose_ids'].squeeze().tolist()
      anchor_nearest_pose_ids = (
          train_data['anchor_nearest_pose_ids'].squeeze().tolist()
      )

      ref_frame_idx = int(train_data['id'].item())
      anchor_frame_idx = int(train_data['anchor_id'].item())

      ref_time_offset = [
          int(near_idx - ref_frame_idx) for near_idx in nearest_pose_ids
      ] 
      anchor_time_offset = [
          int(near_idx - anchor_frame_idx)
          for near_idx in anchor_nearest_pose_ids
      ]
      num_dy_views = len(ref_time_offset) + args.num_vv  # hard-code here!

      # load training rays
      ray_sampler = RaySamplerSingleImage(train_data, device)
      n_rand = int(1.0 * args.N_rand)

      ray_batch = ray_sampler.random_sample(
          n_rand,
          sample_mode=args.sample_mode,
      )

      cb_src_rgbs = torch.cat(
          [
              ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2),
              ray_batch['anchor_src_rgbs'].squeeze(0).permute(0, 3, 1, 2),
          ],
          dim=0,
      )

      cb_featmaps_1, _ = model.feature_net(cb_src_rgbs)
      ref_featmaps, anchor_featmaps = (
          cb_featmaps_1[0:num_dy_views],
          cb_featmaps_1[num_dy_views:],
      )

      static_src_rgbs = (
          ray_batch['static_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
      )
      static_featmaps_coarse, _ = model.feature_net_st(static_src_rgbs)

      ret = render_rays_mono(
          frame_idx=(ref_frame_idx, anchor_frame_idx),
          time_embedding=(ref_time_embedding, anchor_time_embedding),
          time_offset=(ref_time_offset, anchor_time_offset),
          ray_batch=ray_batch,
          model=model,
          projector=projector,
          featmaps=(ref_featmaps, anchor_featmaps, static_featmaps_coarse),
          N_samples=args.N_samples,
          args=args,
          inv_uniform=args.inv_uniform,
          N_importance=args.N_importance,
          det=args.det,
          white_bkgd=args.white_bkgd,
          is_train=False,
          num_vv=args.num_vv
      )

      # # compute loss for static region only
      model.optimizer.zero_grad()
      static_static_mask = 1.0 - ray_batch['static_mask'].float()
      static_static_mask *= ret['outputs_coarse_ref']['mask'].float()

      static_loss = compute_rgb_loss(
          ret['outputs_coarse_st']['rgb'], ray_batch, static_static_mask
      )

      loss = static_loss

      loss.backward()
      model.optimizer.step()
      global_step += 1

      if global_step % args.i_img == 0:
        print('Logging current training view...')
        tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device)
        H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
        gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
        gt_disp = tmp_ray_train_sampler.disp.reshape(H, W, 1)
        log_view_to_tb(
            writer,
            global_step,
            args,
            num_dy_views,
            model,
            tmp_ray_train_sampler,
            projector,
            gt_img,
            gt_disp,
            frame_idx=(ref_frame_idx, anchor_frame_idx),
            time_embedding=(ref_time_embedding, anchor_time_embedding),
            time_offset=(ref_time_offset, anchor_time_offset),
            render_stride=1,
            prefix='train/',
        )

        torch.cuda.empty_cache()

  for epoch in range(start_epoch, int(10**5)):
    if global_step > model.start_step + args.n_iters + 1:
      break

    train_dataset.set_epoch(epoch)
    print('====================================== ', epoch)

    for ii, train_data in enumerate(train_loader):
      time0 = time.time()
      ref_time_embedding = train_data['ref_time'].to(device)
      anchor_time_embedding = train_data['anchor_time'].to(device)

      nearest_pose_ids = train_data['nearest_pose_ids'].squeeze().tolist()
      anchor_nearest_pose_ids = (
          train_data['anchor_nearest_pose_ids'].squeeze().tolist()
      )

      ref_frame_idx = int(train_data['id'].item())
      anchor_frame_idx = int(train_data['anchor_id'].item())

      ref_time_offset = [
          int(near_idx - ref_frame_idx) for near_idx in nearest_pose_ids
      ]
      anchor_time_offset = [
          int(near_idx - anchor_frame_idx)
          for near_idx in anchor_nearest_pose_ids
      ]
      num_dy_views = len(ref_time_offset) + args.num_vv # hard-code here!

      # load training rays
      ray_sampler = RaySamplerSingleImage(train_data, device)
      n_rand = int(1.0 * args.N_rand)
      ray_batch = ray_sampler.random_sample(
          n_rand,
          sample_mode=args.sample_mode,
      )

      cb_src_rgbs = torch.cat(
          [
              ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2),
              ray_batch['anchor_src_rgbs'].squeeze(0).permute(0, 3, 1, 2),
          ],
          dim=0,
      )

      cb_featmaps_1, _ = model.feature_net(cb_src_rgbs)
      ref_featmaps, anchor_featmaps = (
          cb_featmaps_1[0:num_dy_views],
          cb_featmaps_1[num_dy_views:],
      )

      static_src_rgbs = (
          ray_batch['static_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
      )
      static_featmaps_coarse, _ = model.feature_net_st(static_src_rgbs)

      ret = render_rays_mono(
          frame_idx=(ref_frame_idx, anchor_frame_idx),
          time_embedding=(ref_time_embedding, anchor_time_embedding),
          time_offset=(ref_time_offset, anchor_time_offset),
          ray_batch=ray_batch,
          model=model,
          projector=projector,
          featmaps=(ref_featmaps, anchor_featmaps, static_featmaps_coarse),
          N_samples=args.N_samples,
          args=args,
          inv_uniform=args.inv_uniform,
          N_importance=args.N_importance,
          det=args.det,
          white_bkgd=args.white_bkgd,
          num_vv=args.num_vv
      )

      # # compute loss
      model.optimizer.zero_grad()
      divisor = epoch // args.init_decay_epoch

      rgb_loss = rgb_criterion(ret['outputs_coarse_ref'], ray_batch)
      rgb_loss += compute_temporal_rgb_loss(
          ret['outputs_coarse_anchor'], ray_batch
      )
      # RGB loss for dynamic regions only
      if epoch < (args.init_decay_epoch):
        dynamic_mask = (
            ret['outputs_coarse_ref']['mask'].float()
            * ray_batch['motion_mask'].float()
        )
        rgb_loss += compute_rgb_loss(
            ret['outputs_coarse_ref']['rgb_dy'], ray_batch, dynamic_mask
        )

      dynamic_rgb_decay_rate = 10.0
      rgb_loss += rgb_criterion(
          ret['outputs_coarse_ref_dy'],
          ray_batch,
          motion_mask=ray_batch['motion_mask'].float(),
      ) / ((dynamic_rgb_decay_rate) ** divisor)
      rgb_loss += compute_temporal_rgb_loss(
          ret['outputs_coarse_anchor_dy'],
          ray_batch,
          motion_mask=ray_batch['motion_mask'].float(),
      ) / ((dynamic_rgb_decay_rate) ** divisor)

      # disparity loss
      w_disp = args.w_disp / (decay_rate**divisor)
      pred_disp = 1.0 / torch.clamp(
          ret['outputs_coarse_ref']['depth'], min=1e-2
      )

      gt_disp = ray_batch['disp']
      pred_mask = ret['outputs_coarse_ref']['mask']
      disp_loss = (
          w_disp
          * torch.sum(torch.abs(pred_disp - gt_disp) * pred_mask)
          / (torch.sum(pred_mask) + 1e-8)
      )

      # # flow loss
      w_flow = args.w_flow / (decay_rate**divisor)
      flow_mask = pred_mask[None, :, None] * ray_batch['masks']
      flow_loss = w_flow * compute_flow_loss(
          ret['outputs_coarse_ref']['render_flows'],
          ray_batch['flows'],
          flow_mask,
      )

      # trajectory consistency loss
      if args.anneal_cycle:
        w_cycle = min(0.5, args.w_cycle + divisor * args.cycle_factor)
      else:
        w_cycle = args.w_cycle

      pts_traj_anchor = ret['outputs_coarse_anchor']['pts_traj_anchor']
      pts_traj_ref = ret['outputs_coarse_anchor']['pts_traj_ref']

      occ_weights = ret['outputs_coarse_anchor']['occ_weights'][
          None, ..., None
      ].repeat(pts_traj_anchor.shape[0], 1, 1, pts_traj_anchor.shape[-1])
      cycle_loss = (
          w_cycle
          * torch.sum(
              torch.abs((pts_traj_ref - pts_traj_anchor)) * occ_weights
          )
          / (torch.sum(occ_weights) + 1e-8)
      )

      # trajectory regularization loss
      w_reg = args.w_reg
      # minimal scene flow loss
      reg_loss = w_reg * torch.mean(
          torch.abs((ret['outputs_coarse_anchor']['sf_seq']))
      )
      # temporal smooth loss
      reg_loss += (
          w_reg
          * 0.5
          * torch.mean(
              torch.pow(
                  ret['outputs_coarse_anchor']['sf_seq'][:-1]
                  - ret['outputs_coarse_anchor']['sf_seq'][1:],
                  2,
              )
          )
      )
      # spatial smooth loss
      reg_loss += w_reg * torch.mean(
          torch.abs(
              ret['outputs_coarse_anchor']['sf_seq'][:, :, 1:, :]
              - ret['outputs_coarse_anchor']['sf_seq'][:, :, :-1, :]
          )
      )

      # weight entropy loss
      render_weights_dy = torch.sum(
          ret['outputs_coarse_ref']['weights_dy'], dim=-1
      )
      render_weights_st = torch.sum(
          ret['outputs_coarse_ref']['weights_st'], dim=-1
      )
      weights_ratio = render_weights_dy / torch.clamp(
          render_weights_dy + render_weights_st, min=1e-9
      )
      entropy_loss = -(
          weights_ratio * torch.log(weights_ratio + 1e-9)
          + (1.0 - weights_ratio) * torch.log(1.0 - weights_ratio + 1e-9)
      )
      entropy_loss = args.w_skew_entropy * torch.mean(entropy_loss)

      # distortion loss used in mip-nerf-360
      s_vals = ret['outputs_coarse_ref']['s_vals']
      mid_dist = (s_vals[:, 1:] + s_vals[:, :-1]) * 0.5
      interval = s_vals[:, 1:] - s_vals[:, :-1]

      w_distortion = args.w_distortion
      distortion_loss = w_distortion * eff_distloss_native(
          ret['outputs_coarse_ref']['weights'][:, :-1], mid_dist, interval
      )

      # adaptive weight based on current esimtate of decompsotion
      static_static_mask = 1.0 - ray_batch['static_mask'].float()
      static_static_mask *= ret['outputs_coarse_ref']['mask'].float()
      static_static_mask *= (1.0 - weights_ratio).float().detach()

      static_loss = compute_rgb_loss(
          ret['outputs_coarse_ref']['rgb_static'],
          ray_batch,
          static_static_mask,
      )

      # Force static region with > 0.9 prob to have zero dynamic weights
      if divisor > 4:
        static_sfm_mask_2 = static_static_mask * (weights_ratio < 0.1).float()
        static_loss += (
            0.1
            * torch.sum(
                torch.abs(render_weights_dy * static_sfm_mask_2.detach())
            )
            / torch.sum(static_sfm_mask_2 + 1e-8)
        )

      loss = (
          rgb_loss
          + cycle_loss
          + flow_loss
          + disp_loss
          + reg_loss
          + entropy_loss
          + distortion_loss
          + static_loss
      )

      scalars_to_log['loss'] = loss.item()
      scalars_to_log['flow_loss'] = flow_loss.item()
      scalars_to_log['disp_loss'] = disp_loss.item()
      scalars_to_log['rgb_loss'] = rgb_loss.item()
      scalars_to_log['distortion_loss'] = distortion_loss.item()
      scalars_to_log['entropy_loss'] = entropy_loss.item()
      scalars_to_log['static_loss'] = static_loss.item()

      loss.backward()
      model.optimizer.step()

      if model.scheduler.get_last_lr()[0] > 5e-7:
        model.scheduler.step()

      scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
      # end of core optimization loop
      dt = time.time() - time0

      if (
          args.local_rank == 0
          and global_step % 10 == 0
          and len(nearest_pose_ids) < 7
      ):
        print('expname ', args.expname)
        print('divisor ', divisor)
        print(
            'disp_loss ', scalars_to_log['disp_loss'],
            ' flow_loss ', scalars_to_log['flow_loss'],
            ' rgb_loss ', scalars_to_log['rgb_loss'],
        )
        print(
            'cycle_loss ', scalars_to_log['rgb_loss'],
            ' entropy_loss ', scalars_to_log['entropy_loss'],
        )
        print(
            'distortion_loss ', scalars_to_log['distortion_loss'],
            ' static_loss ', scalars_to_log['static_loss']
        )
        print(' divisor ', divisor)  # , ' var_reg_loss ', var_reg_loss.item())
        print(
            'epoch %d global_step %d' % (epoch, global_step),
            ' dt optimization ',
            dt,
        )

      if epoch + 1 == args.init_decay_epoch * 5:
        fpath = os.path.join(out_folder, 'model_no-vv.pth')
        if not os.path.exists(fpath):
          model.save_model(fpath, global_step)

      # Rest is logging
      if args.local_rank == 0:
        if global_step % args.i_print == 0:
          # write mse and psnr stats
          mse_error = img2mse(
              ret['outputs_coarse_ref']['rgb'], ray_batch['rgb']
          ).item()
          scalars_to_log['train/coarse-loss'] = mse_error
          scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(
              mse_error
          )
          if ret['outputs_fine'] is not None:
            mse_error = img2mse(
                ret['outputs_fine']['rgb'], ray_batch['rgb']
            ).item()
            scalars_to_log['train/fine-loss'] = mse_error
            scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(
                mse_error
            )

          logstr = '{} Epoch: {}  step: {} '.format(
              args.expname, epoch, global_step
          )
          for k in scalars_to_log:
            logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
            writer.add_scalar(k, scalars_to_log[k], global_step)
          print(logstr)
          print('each iter time {:.05f} seconds'.format(dt))

        if global_step % args.i_weights == 0:
          print(
              'Saving checkpoints at {} to {}...'.format(
                  global_step, out_folder
              )
          )
          fpath = os.path.join(
              out_folder, 'model_latest.pth'.format(global_step)
          )
          model.save_model(fpath, global_step)

        if global_step % args.i_img == 0:
          print('Logging current training view...')
          tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device)
          H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
          gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
          gt_disp = tmp_ray_train_sampler.disp.reshape(H, W, 1)
          log_view_to_tb(
              writer,
              global_step,
              args,
              num_dy_views,
              model,
              tmp_ray_train_sampler,
              projector,
              gt_img,
              gt_disp,
              frame_idx=(ref_frame_idx, anchor_frame_idx),
              time_embedding=(ref_time_embedding, anchor_time_embedding),
              time_offset=(ref_time_offset, anchor_time_offset),
              render_stride=1,
              prefix='train/',
          )

          torch.cuda.empty_cache()

      global_step += 1


def log_view_to_tb(
    writer,
    global_step,
    args,
    num_dy_views,
    model,
    ray_sampler,
    projector,
    gt_img,
    gt_disp,
    frame_idx,
    time_embedding,
    time_offset,
    render_stride=1,
    prefix='',
):
  """Log rendered images to tensorboard.

  Args:
    writer: tensorboard writter
    global_step: global step
    args: arguments list
    num_dy_views: number of source views for dynamic model
    model: Dynibar Model
    ray_sampler: ray sampelr module
    projector: projection module
    gt_img: ground truth image
    gt_disp: ground truth disparity 
    frame_idx: video frame index
    time_embedding: time embeeding 
    time_offset: offset w.r.t reference time
    render_stride: rendering every x pixel
    prefix: prefix for tensorboard text
  """
  model.switch_to_eval()
  with torch.no_grad():
    ray_batch = ray_sampler.get_all()
    if model.feature_net is not None:
      cb_src_rgbs = torch.cat(
          [
              ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2),
              ray_batch['anchor_src_rgbs'].squeeze(0).permute(0, 3, 1, 2),
          ],
          dim=0,
      )

      cb_featmaps_1, _ = model.feature_net(cb_src_rgbs)
      ref_featmaps, anchor_featmaps = (
          cb_featmaps_1[0:num_dy_views],
          cb_featmaps_1[num_dy_views:],
      )

      static_src_rgbs = (
          ray_batch['static_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
      )
      static_featmaps, _ = model.feature_net_st(static_src_rgbs)

      featmaps = (ref_featmaps, anchor_featmaps, static_featmaps)
    else:
      featmaps = [None, None]

    ret = render_single_image_mono(
        frame_idx=frame_idx,
        time_embedding=time_embedding,
        time_offset=time_offset,
        ray_sampler=ray_sampler,
        ray_batch=ray_batch,
        model=model,
        projector=projector,
        chunk_size=args.chunk_size,
        N_samples=args.N_samples,
        args=args,
        inv_uniform=args.inv_uniform,
        det=True,
        N_importance=args.N_importance,
        white_bkgd=args.white_bkgd,
        render_stride=render_stride,
        featmaps=featmaps,
        num_vv=args.num_vv
    )

  rgb_gt = img_HWC2CHW(gt_img)
  ref_rgb_pred = img_HWC2CHW(ret['outputs_coarse_ref']['rgb'].detach().cpu())
  static_rgb_pred = img_HWC2CHW(
      ret['outputs_coarse_ref']['rgb_static'].detach().cpu()
  )
  dy_rgb_pred = img_HWC2CHW(ret['outputs_coarse_ref']['rgb_dy'].detach().cpu())

  anchor_rgb_pred = img_HWC2CHW(
      ret['outputs_coarse_anchor']['rgb'].detach().cpu()
  )
  st_rgb_pred = img_HWC2CHW(ret['outputs_coarse_st']['rgb'].detach().cpu())

  gt_flows = ray_batch['flows'].reshape(
      ray_batch['flows'].shape[0], gt_img.shape[0], gt_img.shape[1], 2
  )

  exp_sf = ret['outputs_coarse_ref']['exp_sf'].detach().cpu()
  exp_sf_mag = torch.norm(exp_sf, dim=-1)

  ref_depth_im = ret['outputs_coarse_ref']['depth'].detach().cpu()
  anchor_depth_im = ret['outputs_coarse_anchor']['depth'].detach().cpu()
  occ_weight_map = ret['outputs_coarse_anchor']['occ_weight_map'].detach().cpu()

  writer.add_image(
      prefix + 'render_rgb_coarse_ref',
      torch.clamp(ref_rgb_pred, 0.0, 1.0),
      global_step,
  )
  writer.add_image(
      prefix + 'render_rgb_coarse_anchor',
      torch.clamp(anchor_rgb_pred, 0.0, 1.0),
      global_step,
  )
  writer.add_image(
      prefix + 'render_rgb_static',
      torch.clamp(static_rgb_pred, 0.0, 1.0),
      global_step,
  )
  writer.add_image(
      prefix + 'render_rgb_dynamic',
      torch.clamp(dy_rgb_pred, 0.0, 1.0),
      global_step,
  )
  writer.add_image(
      prefix + 'st_rgb_pred', torch.clamp(st_rgb_pred, 0.0, 1.0), global_step
  )

  render_depth_ref = img_HWC2CHW(
      colorize(ref_depth_im, cmap_name='jet', append_cbar=False)
  )

  occ_weight_map_viz = img_HWC2CHW(
      colorize(occ_weight_map, cmap_name='gray', append_cbar=False)
  )

  gt_disp_viz = img_HWC2CHW(
      colorize(gt_disp[..., 0], cmap_name='jet', append_cbar=False)
  )
  exp_sf_mag = img_HWC2CHW(
      colorize(exp_sf_mag, cmap_name='gray', append_cbar=False)
  )

  writer.add_image(
      prefix + 'render_depth_coarse', render_depth_ref, global_step
  )

  writer.add_image(prefix + 'occ_weight_map', occ_weight_map_viz, global_step)
  writer.add_image(prefix + 'exp_sf_mag', exp_sf_mag, global_step)

  writer.add_image(prefix + 'gt_disp_coarse', gt_disp_viz, global_step)
  writer.add_image(prefix + 'gt_rgb_coarse', rgb_gt, global_step)

  # write flow
  rd_flow_stack = []
  gt_flow_stack = []
  for ii in range(min(6, gt_flows.shape[0])):
    rd_flow_stack.append(
        torch.Tensor(
            flow_to_image(
                ret['outputs_coarse_ref']['render_flows'][ii].cpu().numpy()
            )
            / 255.0
        )
    )
    gt_flow_stack.append(
        torch.Tensor(flow_to_image(gt_flows[ii].cpu().numpy()) / 255.0)
    )

  rd_flow_stack = torch.stack(rd_flow_stack, dim=0)
  gt_flow_stack = torch.stack(gt_flow_stack, dim=0)

  writer.add_images(
      prefix + 'rd_flow_stack',
      rd_flow_stack,
      global_step=global_step,
      dataformats='NHWC',
  )
  writer.add_images(
      prefix + 'gt_flow_stack',
      gt_flow_stack,
      global_step=global_step,
      dataformats='NHWC',
  )

  model.switch_to_train()
  return


if __name__ == '__main__':
  parser = config.config_parser()
  args = parser.parse_args()

  if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    synchronize()

  train(args)
