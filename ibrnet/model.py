"""Main Dynibar model class definition."""


import os
from ibrnet.feature_network import ResNet
from ibrnet.mlp_network import DynibarDynamic
from ibrnet.mlp_network import DynibarStatic
from ibrnet.mlp_network import MotionMLP
import numpy as np
import torch


def de_parallel(model):
  """convert distributed parallel model to single model."""
  return model.module if hasattr(model, 'module') else model


def init_dct_basis(num_basis, num_frames):
  """Initialize motion basis with DCT coefficient."""
  T = num_frames
  K = num_basis
  dct_basis = torch.zeros([T, K])

  for t in range(T):
    for k in range(1, K + 1):
      dct_basis[t, k - 1] = np.sqrt(2.0 / T) * np.cos(
          np.pi / (2.0 * T) * (2 * t + 1) * k
      )

  return dct_basis


class DynibarFF(object):
  """Dynibar model for forward-facing benchmark."""

  def __init__(self, args, load_opt=True, load_scheduler=True):
    self.args = args
    self.device = torch.device('cuda:{}'.format(args.local_rank))
    # create coarse DynIBaR models
    self.net_coarse_st = DynibarStatic(
        args,
        in_feat_ch=self.args.coarse_feat_dim,
        n_samples=self.args.N_samples,
    ).to(self.device)
    self.net_coarse_dy = DynibarDynamic(
        args,
        in_feat_ch=self.args.coarse_feat_dim,
        n_samples=self.args.N_samples,
    ).to(self.device)

    # create fine DynIBaR models
    self.net_fine_st = DynibarStatic(
        args,
        in_feat_ch=self.args.fine_feat_dim,
        n_samples=self.args.N_samples + self.args.N_importance,
    ).to(self.device)
    self.net_fine_dy = DynibarDynamic(
        args,
        in_feat_ch=self.args.fine_feat_dim,
        n_samples=self.args.N_samples + self.args.N_importance,
    ).to(self.device)

    # create coarse feature extraction network
    self.feature_net = ResNet(
        coarse_out_ch=self.args.coarse_feat_dim,
        fine_out_ch=self.args.fine_feat_dim,
        coarse_only=False,
    ).to(self.device)

    # create fine feature extraction network
    self.feature_net_fine = ResNet(
        coarse_out_ch=self.args.coarse_feat_dim,
        fine_out_ch=self.args.fine_feat_dim,
        coarse_only=False,
    ).to(self.device)

    # Motion trajectory models with MLPs
    self.motion_mlp = (
        MotionMLP(num_basis=args.num_basis).float().to(self.device)
    )
    self.motion_mlp_fine = (
        MotionMLP(num_basis=args.num_basis).float().to(self.device)
    )

    # Motion basis
    dct_basis = init_dct_basis(args.num_basis, args.num_frames)
    self.trajectory_basis = (
        torch.nn.parameter.Parameter(dct_basis)
        .float()
        .to(self.device)
        .detach()
        .requires_grad_(True)
    )
    self.trajectory_basis_fine = (
        torch.nn.parameter.Parameter(dct_basis)
        .float()
        .to(self.device)
        .detach()
        .requires_grad_(True)
    )

    self.load_coarse_from_ckpt(args.coarse_dir)

    out_folder = os.path.join(args.rootdir, 'checkpoints/fine', args.expname)

    self.optimizer = torch.optim.Adam([
        {
            'params': self.net_fine_st.parameters(),
            'lr': args.lrate_mlp * args.lr_multipler,
        },
        {'params': self.net_fine_dy.parameters(), 'lr': args.lrate_mlp},
        {
            'params': self.feature_net_fine.parameters(),
            'lr': args.lrate_feature,
        },
        {'params': self.motion_mlp_fine.parameters(), 'lr': args.lrate_mlp},
        {'params': self.trajectory_basis_fine, 'lr': args.lrate_mlp * 0.25},
    ])

    self.scheduler = torch.optim.lr_scheduler.StepLR(
        self.optimizer,
        step_size=args.lrate_decay_steps,
        gamma=args.lrate_decay_factor,
    )

    self.start_step = self.load_fine_from_ckpt(
        out_folder, load_opt=True, load_scheduler=True
    )

    device_ids = list(range(torch.cuda.device_count()))

    # convert single model to
    # multi-GPU distributed mode for coarse networks
    self.net_coarse_st = torch.nn.DataParallel(
        self.net_coarse_st, device_ids=device_ids
    )
    self.net_coarse_dy = torch.nn.DataParallel(
        self.net_coarse_dy, device_ids=device_ids
    )
    self.feature_net = torch.nn.DataParallel(
        self.feature_net, device_ids=device_ids
    )
    self.motion_mlp = torch.nn.DataParallel(
        self.motion_mlp, device_ids=device_ids
    )
    # convert single model to
    # multi-GPU distributed mode for fine networks
    self.net_fine_st = torch.nn.DataParallel(
        self.net_fine_st, device_ids=device_ids
    )
    self.net_fine_dy = torch.nn.DataParallel(
        self.net_fine_dy, device_ids=device_ids
    )
    self.feature_net_fine = torch.nn.DataParallel(
        self.feature_net_fine, device_ids=device_ids
    )
    self.motion_mlp_fine = torch.nn.DataParallel(
        self.motion_mlp_fine, device_ids=device_ids
    )

  def switch_to_eval(self):
    """Switch to evaluation model."""
    self.net_fine_st.eval()
    self.net_fine_dy.eval()

    self.feature_net_fine.eval()
    self.motion_mlp_fine.eval()

  def switch_to_train(self):
    """Switch to training model."""
    self.net_fine_st.train()
    self.net_fine_dy.train()

    self.feature_net_fine.train()
    self.motion_mlp_fine.train()

  def save_model(self, filename, global_step):
    """De-parallel and save current model to local disk."""
    to_save = {
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'net_fine_st': de_parallel(self.net_fine_st).state_dict(),
        'net_fine_dy': de_parallel(self.net_fine_dy).state_dict(),
        'feature_net_fine': de_parallel(self.feature_net_fine).state_dict(),
        'motion_mlp_fine': de_parallel(self.motion_mlp_fine).state_dict(),
        'traj_basis_fine': self.trajectory_basis_fine,
        'global_step': int(global_step),
    }

    torch.save(to_save, filename)

  def load_coarse_model(self, filename):
    """Load coarse stage dynibar model."""
    if self.args.distributed:
      to_load = torch.load(
          filename, map_location='cuda:{}'.format(self.args.local_rank)
      )
    else:
      to_load = torch.load(filename)

    self.net_coarse_st.load_state_dict(to_load['net_coarse_st'])
    self.net_coarse_dy.load_state_dict(to_load['net_coarse_dy'])

    self.feature_net.load_state_dict(to_load['feature_net'])

    self.motion_mlp.load_state_dict(to_load['motion_mlp'])
    self.trajectory_basis = to_load['traj_basis']

    return to_load['global_step']

  def load_fine_model(self, filename, load_opt=True, load_scheduler=True):
    """Load fine stage dynibar model."""
    if self.args.distributed:
      to_load = torch.load(
          filename, map_location='cuda:{}'.format(self.args.local_rank)
      )
    else:
      to_load = torch.load(filename)

    if load_opt:
      self.optimizer.load_state_dict(to_load['optimizer'])
    if load_scheduler:
      self.scheduler.load_state_dict(to_load['scheduler'])

    self.net_fine_st.load_state_dict(to_load['net_fine_st'])
    self.net_fine_dy.load_state_dict(to_load['net_fine_dy'])

    self.feature_net_fine.load_state_dict(to_load['feature_net_fine'])

    self.motion_mlp_fine.load_state_dict(to_load['motion_mlp_fine'])
    self.trajectory_basis_fine = to_load['traj_basis_fine']

    return to_load['global_step']

  def load_coarse_from_ckpt(
      self,
      out_folder
  ):
    """Load coarse model from existing checkpoints and return the current step."""

    # all existing ckpts
    ckpts = []
    if os.path.exists(out_folder):
      ckpts = [
          os.path.join(out_folder, f)
          for f in sorted(os.listdir(out_folder))
          if f.endswith('.pth')
      ]

    fpath = ckpts[-1]
    num_steps = self.load_coarse_model(fpath)

    step = num_steps
    print('Reloading from {}, starting at step={}'.format(fpath, step))

    return step

  def load_fine_from_ckpt(
      self,
      out_folder,
      load_opt=True,
      load_scheduler=True
  ):
    """Load fine model from existing checkpoints and return the current step."""

    # all existing ckpts
    ckpts = []
    if os.path.exists(out_folder):
      ckpts = [
          os.path.join(out_folder, f)
          for f in sorted(os.listdir(out_folder))
          if f.endswith('.pth')
      ]

    if self.args.ckpt_path is not None:
      if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
        ckpts = [self.args.ckpt_path]

    if len(ckpts) > 0 and not self.args.no_reload:
      fpath = ckpts[-1]
      num_steps = self.load_fine_model(fpath, load_opt, load_scheduler)
      step = num_steps
      print('Reloading from {}, starting at step={}'.format(fpath, step))
    else:
      print('No ckpts found, training from scratch...')
      step = 0

    return step


class DynibarMono(object):
  """Main Dynibar model for monocular video."""

  def __init__(self, args):
    self.args = args
    self.device = torch.device('cuda:{}'.format(args.local_rank))
    # create Dynibar models for monocular videos
    self.net_coarse_st = DynibarStatic(
        args,
        in_feat_ch=self.args.coarse_feat_dim,
        n_samples=self.args.N_samples,
    ).to(self.device)
    self.net_coarse_dy = DynibarDynamic(
        args,
        in_feat_ch=self.args.coarse_feat_dim,
        n_samples=self.args.N_samples,
        shift=5.0,
    ).to(self.device)

    self.net_fine = None

    # create feature extraction network used for dynamic model.
    self.feature_net = ResNet(
        coarse_out_ch=self.args.coarse_feat_dim,
        fine_out_ch=self.args.fine_feat_dim,
        coarse_only=False,
    ).to(self.device)

    # create feature extraction network used for static model.
    self.feature_net_st = ResNet(
        coarse_out_ch=self.args.coarse_feat_dim,
        fine_out_ch=self.args.fine_feat_dim,
        coarse_only=False,
    ).to(self.device)

    # Motion trajectory model with MLP.
    self.motion_mlp = (
        MotionMLP(num_basis=args.num_basis).float().to(self.device)
    )

    # basis
    dct_basis = init_dct_basis(args.num_basis, args.num_frames)
    self.trajectory_basis = (
        torch.nn.parameter.Parameter(dct_basis)
        .float()
        .to(self.device)
        .detach()
        .requires_grad_(True)
    )

    self.optimizer = torch.optim.Adam([
        {'params': self.net_coarse_st.parameters(), 'lr': args.lrate_mlp * 0.5},
        {
            'params': self.feature_net_st.parameters(),
            'lr': args.lrate_feature * 0.5,
        },
        {'params': self.net_coarse_dy.parameters(), 'lr': args.lrate_mlp},
        {'params': self.feature_net.parameters(), 'lr': args.lrate_feature},
        {'params': self.motion_mlp.parameters(), 'lr': args.lrate_mlp},
        {'params': self.trajectory_basis, 'lr': args.lrate_mlp * 0.25},
    ])

    print(
        'lrate_decay_steps ',
        args.lrate_decay_steps,
        ' lrate_decay_factor ',
        args.lrate_decay_factor,
    )

    self.scheduler = torch.optim.lr_scheduler.StepLR(
        self.optimizer,
        step_size=args.lrate_decay_steps,
        gamma=args.lrate_decay_factor,
    )

    out_folder = os.path.join(args.rootdir, 'out', args.expname)

    self.start_step = 0

    if args.pretrain_path == '':
      self.start_step = self.load_from_ckpt(
          out_folder, load_opt=True, load_scheduler=True
      )

    else:
      self.start_step = self.load_from_ckpt(
          args.pretrain_path, load_opt=True, load_scheduler=True
      )

    device_ids = list(range(torch.cuda.device_count()))

    self.net_coarse_st = torch.nn.DataParallel(
        self.net_coarse_st, device_ids=device_ids
    )
    self.net_coarse_dy = torch.nn.DataParallel(
        self.net_coarse_dy, device_ids=device_ids
    )
    self.feature_net = torch.nn.DataParallel(
        self.feature_net, device_ids=device_ids
    )
    self.feature_net_st = torch.nn.DataParallel(
        self.feature_net_st, device_ids=device_ids
    )

    self.motion_mlp = torch.nn.DataParallel(
        self.motion_mlp, device_ids=device_ids
    )

  def switch_to_eval(self):
    """Switch models to evaluation mode."""
    self.net_coarse_st.eval()
    self.net_coarse_dy.eval()

    self.feature_net.eval()
    self.feature_net_st.eval()
    self.motion_mlp.eval()

    if self.net_fine is not None:
      self.net_fine.eval()

  def switch_to_train(self):
    """Switch models to training mode."""

    self.net_coarse_st.train()
    self.net_coarse_dy.train()

    self.feature_net.train()
    self.motion_mlp.train()
    self.feature_net_st.train()

    if self.net_fine is not None:
      self.net_fine.train()

  def save_model(self, filename, global_step):
    """Save Dynibar monocular model."""
    to_save = {
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'net_coarse_st': de_parallel(self.net_coarse_st).state_dict(),
        'net_coarse_dy': de_parallel(self.net_coarse_dy).state_dict(),
        'feature_net': de_parallel(self.feature_net).state_dict(),
        'feature_net_st': de_parallel(self.feature_net_st).state_dict(),
        'motion_mlp': de_parallel(self.motion_mlp).state_dict(),
        'traj_basis': self.trajectory_basis,
        'global_step': int(global_step),
    }

    if self.net_fine is not None:
      to_save['net_fine'] = de_parallel(self.net_fine).state_dict()

    torch.save(to_save, filename)

  def load_model(self, filename, load_opt=True, load_scheduler=True):
    """Load Dynibar monocular model."""
    if self.args.distributed:
      to_load = torch.load(
          filename, map_location='cuda:{}'.format(self.args.local_rank)
      )
    else:
      to_load = torch.load(filename)

    if load_opt:
      self.optimizer.load_state_dict(to_load['optimizer'])
    if load_scheduler:
      self.scheduler.load_state_dict(to_load['scheduler'])

    self.net_coarse_st.load_state_dict(to_load['net_coarse_st'])
    self.net_coarse_dy.load_state_dict(to_load['net_coarse_dy'])

    self.feature_net.load_state_dict(to_load['feature_net'])
    self.feature_net_st.load_state_dict(to_load['feature_net_st'])

    self.motion_mlp.load_state_dict(to_load['motion_mlp'])
    self.trajectory_basis = to_load['traj_basis']

    return to_load['global_step']

  def load_from_ckpt(
      self,
      out_folder,
      load_opt=True,
      load_scheduler=True,
  ):
    """Load coarse model from existing checkpoints and return the current step."""

    # all existing ckpts
    ckpts = []
    if os.path.exists(out_folder):
      ckpts = [
          os.path.join(out_folder, f)
          for f in sorted(os.listdir(out_folder))
          if f.endswith('latest.pth')
      ]

    if self.args.ckpt_path is not None:
      if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
        ckpts = [self.args.ckpt_path]

    if len(ckpts) > 0 and not self.args.no_reload:
      fpath = ckpts[-1]
      num_steps = self.load_model(fpath, True, True)
      print('=========== num_steps ', num_steps)

      step = num_steps
      print('Reloading from {}, starting at step={}'.format(fpath, step))
    else:
      print('No ckpts found, training from scratch...')
      step = 0

    return step

