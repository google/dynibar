"""function definition for config function."""

import configargparse


def config_parser():
  """Configuration function."""
  parser = configargparse.ArgumentParser()
  # general
  parser.add_argument('--config', is_config_file=True, help='Config file path')
  parser.add_argument(
      '--rootdir',
      type=str,
      help=(
          'The path to the project root directory. Replace this path with'
          ' yours!'
      ),
  )
  parser.add_argument(
      '--folder_path',
      type=str,
      help=(
          'The path to the input training data. Replace this path with yours.'
      ),
  )

  parser.add_argument(
      '--coarse_dir',
      type=str,
      help=(
          'The directory of coarse model.'
      ),
  )

  parser.add_argument(
      '--mask_src_view',
      action='store_true',
      help=(
          'Using motion segementation to mask src views for rendering static'
          ' model'
      ),
  )
  parser.add_argument(
      '--training_height', type=int, default=288, help='Training image height'
  )
  parser.add_argument('--expname', type=str, help='Experiment name')
  parser.add_argument(
      '--distributed', action='store_true', help='Use distributed training'
  )
  parser.add_argument(
      '--local_rank', type=int, default=0, help='Rank for distributed training'
  )
  parser.add_argument(
      '-j',
      '--workers',
      default=16,
      type=int,
      help='Number of data loading workers (default: 16)',
  )

  parser.add_argument(
      '--mask_static',
      action='store_true',
      help='Using motion mask to mask source views for static model',
  )

  ########## model options ##########
  parser.add_argument(
      '--N_rand',
      type=int,
      default=32 * 16,
      help='Batch size (number of random rays per gradient step)',
  )
  parser.add_argument(
      '--sample_mode',
      type=str,
      default='uniform',
      help='How to sample pixels from images for training:uniform|center',
  )
  parser.add_argument(
      '--lr_multipler',
      type=float,
      default=1.0,
      help='Learning rate ratio for training static component',
  )
  parser.add_argument(
      '--num_vv',
      type=int,
      default=3,
      help='Number of virtual source views',
  )
  parser.add_argument(
      '--cycle_factor',
      type=float,
      default=0.1,
      help='Cycle conssitency loss warmup factor',
  )
  parser.add_argument(
      '--anneal_cycle',
      action='store_true',
      help='Bootstrap cycle consistency loss',
  )
  parser.add_argument(
      '--erosion_radius',
      type=int,
      default=1,
      help='Mophorlogical erosion raidus for mask',
  )
  parser.add_argument(
      '--decay_rate',
      type=float,
      default=10.0,
      help='Decaying rate for data-driven loss',
  )

  ########## dataset options ##########
  parser.add_argument(
      '--eval_dataset',
      type=str,
      default='llff_test',
      help='The dataset to evaluate',
  )
  parser.add_argument(
      '--eval_scenes',
      nargs='+',
      default=[],
      help='Optional, specify a subset of scenes from eval_dataset to evaluate',
  )
  parser.add_argument(
      '--render_idx', type=int, default=-1, help='Frame index for rendering'
  )
  parser.add_argument(
      '--train_dataset',
      type=str,
      default='ibrnet_collected',
      help=(
          'the training dataset, should either be a single dataset, or multiple'
          ' datasets connected with "+", for example,'
          ' ibrnet_collected+llff+spaces'
      ),
  )
  parser.add_argument(
      '--train_scenes',
      nargs='+',
      default=[],
      help=(
          'optional, specify a subset of training scenes from training dataset'
      ),
  )

  ## others
  parser.add_argument(
      '--init_decay_epoch',
      type=int,
      default=150,
      help='How many epochs to decay data driven losses',
  )
  parser.add_argument(
      '--max_range',
      type=int,
      default=35,
      help='Max frame range to sample source views for static model',
  )

  ########## model options ##########
  ## ray sampling options
  parser.add_argument(
      '--chunk_size',
      type=int,
      default=1024 * 4,
      help=(
          'Number of rays processed in parallel, decrease if running out of'
          ' memory'
      ),
  )
  ## model options
  parser.add_argument(
      '--coarse_feat_dim',
      type=int,
      default=32,
      help='2D feature dimension for coarse level',
  )
  parser.add_argument(
      '--fine_feat_dim',
      type=int,
      default=32,
      help='2D feature dimension for fine level',
  )
  parser.add_argument(
      '--num_source_views',
      type=int,
      default=7,
      help=(
          'The number of input source views for each target view used in'
          'static dynibar model'
      ),
  )
  parser.add_argument(
      '--num_basis',
      type=int,
      default=6,
      help='The number of basis for motion trajectory',
  )
  parser.add_argument(
      '--anti_alias_pooling',
      type=int,
      default=1,
      help='Use anti-alias pooling',
  )
  parser.add_argument(
      '--mask_rgb',
      type=int,
      default=1,
      help=(
          'Mask RGB features coresponding to black pixel for rendering from'
          ' static model'
      ),
  )

  ########## checkpoints ##########
  parser.add_argument(
      '--no_reload',
      action='store_true',
      help='do not reload weights from saved ckpt',
  )
  parser.add_argument(
      '--ckpt_path',
      type=str,
      default='',
      help='specific weights npy file to reload for coarse network',
  )
  parser.add_argument(
      '--no_load_opt',
      action='store_true',
      help='do not load optimizer when reloading',
  )
  parser.add_argument(
      '--no_load_scheduler',
      action='store_true',
      help='do not load scheduler when reloading',
  )
  ########### iterations & learning rate options ##########
  parser.add_argument(
      '--n_iters', type=int, default=300000, help='Num of iterations'
  )
  parser.add_argument(
      '--lrate_feature',
      type=float,
      default=1e-3,
      help='Learning rate for feature extractor',
  )
  parser.add_argument(
      '--lrate_mlp', type=float, default=5e-4, help='Learning rate for mlp'
  )
  parser.add_argument(
      '--lrate_decay_factor',
      type=float,
      default=0.5,
      help='Decay learning rate by a factor every specified number of steps',
  )
  parser.add_argument(
      '--lrate_decay_steps',
      type=int,
      default=50000,
      help='Decay learning rate by a factor every number of steps',
  )
  parser.add_argument(
      '--w_cycle',
      type=float,
      default=0.1,
      help='Weight of cycle consistency loss',
  )
  parser.add_argument(
      '--w_distortion',
      type=float,
      default=1e-3,
      help='Weight of distortion loss',
  )
  parser.add_argument(
      '--w_entropy', type=float, default=0.0, help='Weight of entropy loss'
  )
  parser.add_argument(
      '--w_disp', type=float, default=5e-2, help='Weight of disparty loss'
  )
  parser.add_argument(
      '--w_flow', type=float, default=5e-3, help='Weight of flow loss'
  )
  parser.add_argument(
      '--w_skew_entropy',
      type=float,
      default=1e-3,
      help='Weight of entropy loss, assuming there is no skewness.',
  )
  parser.add_argument(
      '--w_reg', type=float, default=0.05, help='Weight of regularization loss'
  )
  parser.add_argument(
      '--pretrain_path', type=str, default='', help='Pretrained model path'
  )
  parser.add_argument(
      '--occ_weights_mode',
      type=int,
      default=0,
      help=(
          'Occlusion weight mode during cross-time rendering. 0: mix two models'
          ' weights. 1: using weight from dynamic model only 2: using weight'
          ' composited from static and dynamic models. '
      ),
  )

  ########## rendering options ##########
  parser.add_argument(
      '--N_samples',
      type=int,
      default=64,
      help='Number of coarse samples per ray',
  )
  parser.add_argument(
      '--N_importance',
      type=int,
      default=64,
      help=(
          'Number of fine samples per ray. total number of samples is the sum'
          ' of coarse plus fine models'
      ),
  )
  parser.add_argument(
      '--inv_uniform',
      action='store_true',
      help='If True, uniformly sample in inverse depth space',
  )
  parser.add_argument(
      '--input_dir',
      action='store_true',
      help='If True, input global directional with positional encoding',
  )
  parser.add_argument(
      '--input_xyz',
      action='store_true',
      help='If True, input global xyz with positional encoding',
  )
  parser.add_argument(
      '--det',
      action='store_true',
      help='Deterministic sampling for coarse and fine samples',
  )
  parser.add_argument(
      '--white_bkgd',
      action='store_true',
      help='Apply the trick to avoid fitting to white background',
  )
  parser.add_argument(
      '--render_stride',
      type=int,
      default=1,
      help='Render with large stride for validation to save time',
  )
  ########## logging/saving options ##########
  parser.add_argument(
      '--i_print', type=int, default=100, help='Frequency of terminal printout'
  )
  parser.add_argument(
      '--i_img',
      type=int,
      default=1000,
      help='Frequency of tensorboard image logging',
  )
  parser.add_argument(
      '--i_weights',
      type=int,
      default=10000,
      help='Frequency of weight ckpt saving',
  )

  return parser
