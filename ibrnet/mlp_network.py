"""Class definition for MLP Network."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class ScaledDotProductAttention(nn.Module):
  """Dot-Product Attention Layer."""

  def __init__(self, temperature, attn_dropout=0.1):
    super().__init__()
    self.temperature = temperature

  def forward(self, q, k, v, mask=None):
    attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

    if mask is not None:
      attn = attn.masked_fill(mask == 0, -1e9)
      # attn = attn * mask

    attn = F.softmax(attn, dim=-1)
    # attn = self.dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)

    return output, attn


class PositionwiseFeedForward(nn.Module):
  """A two-feed-forward-layer module."""

  def __init__(self, d_in, d_hid, dropout=0.1):
    super().__init__()
    self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
    self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
    self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    # self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    residual = x

    x = self.w_2(F.relu(self.w_1(x)))
    # x = self.dropout(x)
    x += residual

    x = self.layer_norm(x)

    return x


class MultiHeadAttention(nn.Module):
  """Multi-Head Attention module."""

  def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    super().__init__()

    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v

    self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
    self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

    # self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def forward(self, q, k, v, mask=None):
    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
    sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

    residual = q

    # Pass through the pre-attention projection: b x lq x (n*dv)
    # Separate different heads: b x lq x n x dv
    q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
    k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
    v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

    # Transpose for attention dot product: b x n x lq x dv
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    if mask is not None:
      mask = mask.unsqueeze(1)  # For head axis broadcasting.

    q, attn = self.attention(q, k, v, mask=mask)

    # Transpose to move the head dimension back: b x lq x n x dv
    # Combine the last two dimensions to concatenate all the heads together
    q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
    q = self.fc(q)
    q += residual

    q = self.layer_norm(q)

    return q, attn


def weights_init(m):
  """Default initialization of linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.kaiming_normal_(m.weight.data)
    if m.bias is not None:
      nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
  mean = torch.sum(x * weight, dim=2, keepdim=True)
  var = torch.sum(weight * (x - mean) ** 2, dim=2, keepdim=True)
  return mean, var


@torch.jit.script
def epipolar_fused_mean_variance(x, weight):
  mean = torch.sum(x * weight, dim=1, keepdim=True)
  var = torch.sum(weight * (x - mean) ** 2, dim=1, keepdim=True)
  return mean, var


class DynibarDynamic(nn.Module):
  """Dynibar time-varying dynamic model."""

  def __init__(self, args, in_feat_ch=32, n_samples=64, shift=0.0, **kwargs):
    super(DynibarDynamic, self).__init__()
    self.args = args
    self.anti_alias_pooling = False  # args.anti_alias_pooling
    self.input_dir = args.input_dir
    self.input_xyz = args.input_xyz

    if self.anti_alias_pooling:
      self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)

    activation_func = nn.ELU(inplace=True)
    self.shift = shift
    t_num_freqs = 10
    self.t_embed = PeriodicEmbed(
        max_freq=t_num_freqs, N_freq=t_num_freqs, linspace=False
    ).float()
    dir_num_freqs = 4
    self.dir_embed = PeriodicEmbed(
        max_freq=dir_num_freqs, N_freq=dir_num_freqs, linspace=False
    ).float()

    pts_num_freqs = 5
    self.pts_embed = PeriodicEmbed(
        max_freq=pts_num_freqs, N_freq=pts_num_freqs, linspace=False
    ).float()

    self.n_samples = n_samples
    self.ray_dir_fc = nn.Sequential(
        nn.Linear(t_num_freqs * 2 + 1, 256),
        activation_func,
        nn.Linear(256, in_feat_ch + 3),
        activation_func,
    )

    self.base_fc = nn.Sequential(
        nn.Linear((in_feat_ch + 3) * 3, 256),
        activation_func,
        nn.Linear(256, 128),
        activation_func,
    )

    self.vis_fc = nn.Sequential(
        nn.Linear(128, 128),
        activation_func,
        nn.Linear(128, 128 + 1),
        activation_func,
    )

    self.vis_fc2 = nn.Sequential(
        nn.Linear(128, 128), activation_func, nn.Linear(128, 1), nn.Sigmoid()
    )

    self.geometry_fc = nn.Sequential(
        nn.Linear(128 * 2 + 1, 256),
        activation_func,
        nn.Linear(256, 128),
        activation_func,
    )

    self.ray_attention = MultiHeadAttention(4, 128, 32, 32)

    num_c_xyz = (pts_num_freqs * 2 + 1) * 3

    self.ref_pts_fc = nn.Sequential(
        nn.Linear(num_c_xyz + 128, 256),
        activation_func,
        nn.Linear(256, 128),
        activation_func,
    )

    self.out_geometry_fc = nn.Sequential(
        nn.Linear(128, 128), activation_func, nn.Linear(128, 1)
    )

    if self.input_dir:
      self.rgb_fc = nn.Sequential(
          nn.Linear(128 + (dir_num_freqs * 2 + 1) * 3, 128),
          activation_func,
          nn.Linear(128, 64),
          activation_func,
          nn.Linear(64, 3),
          nn.Sigmoid(),
      )
    else:
      raise NotImplementedError

    self.pos_encoding = self.posenc(d_hid=128, n_samples=self.n_samples)

  def posenc(self, d_hid, n_samples):
    def get_position_angle_vec(position):
      return [
          position / np.power(10000, 2 * (hid_j // 2) / d_hid)
          for hid_j in range(d_hid)
      ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_samples)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    sinusoid_table = torch.from_numpy(sinusoid_table).float().unsqueeze(0)
    return sinusoid_table

  def forward(
      self, pts_xyz, rgb_feat, glb_ray_dir, ray_diff, time_diff, mask, time
  ):
    num_views = rgb_feat.shape[2]
    time_pe = (
        self.t_embed(time)[..., None, :].repeat(1, 1, num_views, 1).float()
    )

    direction_feat = self.ray_dir_fc(time_pe)

    # rgb_in = rgb_feat[..., :3]
    rgb_feat = rgb_feat + direction_feat

    if self.anti_alias_pooling:
      _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
      exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
      weight = (
          exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]
      ) * mask
      weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
    else:
      weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

    # compute mean and variance across different views for each point
    mean, var = fused_mean_variance(
        rgb_feat, weight
    )  # [n_rays, n_samples, 1, n_feat]
    globalfeat = torch.cat(
        [mean, var], dim=-1
    )  # [n_rays, n_samples, 1, 2*n_feat]

    x = torch.cat(
        [globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1
    )  # [n_rays, n_samples, n_views, 3*n_feat]
    x = self.base_fc(x)

    x_vis = self.vis_fc(x * weight)
    x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
    vis = F.sigmoid(vis) * mask
    x = x + x_res
    vis = self.vis_fc2(x * vis) * mask
    weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

    mean, var = fused_mean_variance(x, weight)
    globalfeat = torch.cat(
        [mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1
    )  # [n_rays, n_samples, 32*2+1]
    globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
    num_valid_obs = torch.sum(mask, dim=2)

    globalfeat = globalfeat + self.pos_encoding.to(globalfeat.device)
    globalfeat, _ = self.ray_attention(
        globalfeat, globalfeat, globalfeat, mask=(num_valid_obs > 1).float()
    )  # [n_rays, n_samples, 16]

    pts_xyz_pe = self.pts_embed(pts_xyz)
    globalfeat = self.ref_pts_fc(torch.cat([globalfeat, pts_xyz_pe], dim=-1))

    sigma = (
        self.out_geometry_fc(globalfeat) - self.shift
    )  # [n_rays, n_samples, 1]
    sigma_out = sigma.masked_fill(
        num_valid_obs < 1, -1e9
    )  # set the sigma of invalid point to zero

    if self.input_dir:
      glb_ray_dir_pe = self.dir_embed(glb_ray_dir).float()
      h = torch.cat(
          [
              globalfeat,
              glb_ray_dir_pe[:, None, :].repeat(1, globalfeat.shape[1], 1),
          ],
          dim=-1,
      )
    else:
      h = globalfeat

    rgb_out = self.rgb_fc(h)
    rgb_out = rgb_out.masked_fill(torch.sum(mask.repeat(1, 1, 1, 3), 2) == 0, 0)
    out = torch.cat([rgb_out, sigma_out], dim=-1)
    return out


class DynibarStatic(nn.Module):
  """Dynibar time-invariant static model."""

  def __init__(self, args, in_feat_ch=32, n_samples=64, **kwargs):
    super(DynibarStatic, self).__init__()
    self.args = args
    self.anti_alias_pooling = args.anti_alias_pooling  # CHECK DISCREPENCY
    self.mask_rgb = args.mask_rgb
    self.input_dir = args.input_dir
    self.input_xyz = args.input_xyz

    if self.anti_alias_pooling:
      self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)

    activation_func = nn.ELU(inplace=True)

    ray_num_freqs = 5
    self.ray_embed = PeriodicEmbed(
        max_freq=ray_num_freqs, N_freq=ray_num_freqs, linspace=False
    )
    pts_num_freqs = 5
    self.pts_embed = PeriodicEmbed(
        max_freq=pts_num_freqs, N_freq=pts_num_freqs, linspace=False
    )

    num_c_xyz = (pts_num_freqs * 2 + 1) * 3
    num_c_ray = (ray_num_freqs * 2 + 1) * 6

    self.n_samples = n_samples

    self.ray_dir_fc = nn.Sequential(
        nn.Linear(4 + num_c_xyz + num_c_ray, 256),
        activation_func,
        nn.Linear(256, in_feat_ch + 3),
    )

    self.ref_feature_fc = nn.Sequential(nn.Linear(num_c_ray, in_feat_ch + 3))

    self.base_fc = nn.Sequential(
        nn.Linear((in_feat_ch + 3) * 6, 256),
        activation_func,
        nn.Linear(256, 128),
        activation_func,
    )

    self.vis_fc = nn.Sequential(
        nn.Linear(128, 128),
        activation_func,
        nn.Linear(128, 128 + 1),
        activation_func,
    )

    self.vis_fc2 = nn.Sequential(
        nn.Linear(128, 128), activation_func, nn.Linear(128, 1), nn.Sigmoid()
    )

    self.geometry_fc = nn.Sequential(
        nn.Linear(128 * 2 + 1, 256),
        activation_func,
        nn.Linear(256, 128),
        activation_func,
    )

    self.ray_attention = MultiHeadAttention(4, 128, 32, 32)
    self.out_geometry_fc = nn.Sequential(
        nn.Linear(128, 128), activation_func, nn.Linear(128, 1)
    )

    if self.input_dir:
      self.rgb_fc = nn.Sequential(
          nn.Linear(128 * 2 + 1 + 4, 128),
          activation_func,
          nn.Linear(128, 64),
          activation_func,
          nn.Linear(64, 1),
      )

    else:
      self.rgb_fc = nn.Sequential(
          nn.Linear(32 + 1, 32),
          activation_func,
          nn.Linear(32, 16),
          activation_func,
          nn.Linear(16, 1),
      )

    self.pos_encoding = self.posenc(d_hid=128, n_samples=self.n_samples)

  def posenc(self, d_hid, n_samples):
    def get_position_angle_vec(position):
      return [
          position / np.power(10000, 2 * (hid_j // 2) / d_hid)
          for hid_j in range(d_hid)
      ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_samples)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    sinusoid_table = torch.from_numpy(sinusoid_table).float().unsqueeze(0)
    return sinusoid_table

  def forward(
      self,
      pts,
      ref_rays_coords,
      src_rays_coords,
      rgb_feat,
      glb_ray_dir,
      ray_diff,
      mask,
  ):
    num_views = rgb_feat.shape[2]
    ref_rays_pe = self.ray_embed(ref_rays_coords)
    src_rays_pe = self.ray_embed(src_rays_coords)
    pts_pe = self.pts_embed(pts)

    ref_features = ref_rays_pe[:, None, None, :].expand(
        -1, src_rays_pe.shape[1], src_rays_pe.shape[2], -1
    )
    src_features = torch.cat(
        [
            pts_pe.unsqueeze(2).expand(-1, -1, src_rays_pe.shape[2], -1),
            src_rays_pe,
        ],
        dim=-1,
    )

    src_feat = self.ray_dir_fc(torch.cat([src_features, ray_diff], dim=-1))
    ref_feat = self.ref_feature_fc(ref_features)

    rgb_in = rgb_feat[..., :3]

    if self.mask_rgb:
      rgb_in_sum = torch.sum(rgb_in, dim=-1, keepdim=True)
      rgb_mask = (rgb_in_sum > 1e-3).float().detach()
      mask = mask * rgb_mask

    rgb_feat = torch.cat([rgb_feat, src_feat * ref_feat], dim=-1)

    if self.anti_alias_pooling:
      _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
      exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
      weight = (
          exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]
      ) * mask
      weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
    else:
      weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

    # compute mean and variance across different views for each point
    mean, var = fused_mean_variance(
        rgb_feat, weight
    )  # [n_rays, n_samples, 1, n_feat]
    globalfeat = torch.cat(
        [mean, var], dim=-1
    )  # [n_rays, n_samples, 1, 2*n_feat]

    x = torch.cat(
        [globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1
    )  # [n_rays, n_samples, n_views, 3*n_feat]

    x = self.base_fc(x)

    x_vis = self.vis_fc(x * weight)
    x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
    vis = F.sigmoid(vis) * mask
    x = x + x_res
    vis = self.vis_fc2(x * vis) * mask
    weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

    mean, var = fused_mean_variance(x, weight)
    globalfeat = torch.cat(
        [mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1
    )  # [n_rays, n_samples, 32*2+1]
    globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
    num_valid_obs = torch.sum(mask, dim=2)

    # globalfeat = globalfeat #+ self.pos_encoding.to(globalfeat.device)
    globalfeat, _ = self.ray_attention(
        globalfeat, globalfeat, globalfeat, mask=(num_valid_obs > 1).float()
    )  # [n_rays, n_samples, 16]
    sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
    sigma_out = sigma.masked_fill(
        num_valid_obs < 1, -1e9
    )  # set the sigma of invalid point to zero

    if self.input_dir:
      x = torch.cat(
          [
              globalfeat[:, :, None, :].expand(-1, -1, x.shape[2], -1),
              x,
              vis,
              ray_diff,
          ],
          dim=-1,
      )
    else:
      x = torch.cat([globalfeat, vis], dim=-1)

    x = self.rgb_fc(x)

    x = x.masked_fill(mask == 0, -1e9)
    blending_weights_valid = F.softmax(x, dim=2)  # color blending
    rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=2)
    out = torch.cat([rgb_out, sigma_out], dim=-1)
    return out


class PeriodicEmbed(nn.Module):
  """Fourier Position encoding module."""

  def __init__(self, max_freq, N_freq, linspace=True):
    """Init function for position encoding.

    Args:
      max_freq: max frequency band
      N_freq: number of frequency
      linspace: linearly spacing or not
    """
    super().__init__()
    self.embed_functions = [torch.cos, torch.sin]
    if linspace:
      self.freqs = torch.linspace(1, max_freq + 1, steps=N_freq)
    else:
      exps = torch.linspace(0, N_freq - 1, steps=N_freq)
      self.freqs = 2**exps

  def forward(self, x):
    output = [x]
    for f in self.embed_functions:
      for freq in self.freqs:
        output.append(f(freq * x))

    return torch.cat(output, -1)


class MotionMLP(nn.Module):
  """Motion trajectory MLP module."""

  def __init__(
      self,
      num_basis=4,
      D=8,
      W=256,
      input_ch=4,
      num_freqs=16,
      skips=[4],
      sf_mag_div=1.0,
  ):
    """Init function for motion MLP.

    Args:
      num_basis: number motion basis
      D: MLP layers
      W: feature dimention of MLP layers
      input_ch: input number of channels
      num_freqs: number of rquency for position encoding
      skips: where to inject skip connection
      sf_mag_div: motion scaling factor
    """
    super(MotionMLP, self).__init__()
    self.D = D
    self.W = W
    self.input_ch = int(input_ch + input_ch * num_freqs * 2)
    self.skips = skips
    self.sf_mag_div = sf_mag_div

    self.xyzt_embed = PeriodicEmbed(max_freq=num_freqs, N_freq=num_freqs)

    self.pts_linears = nn.ModuleList(
        [nn.Linear(self.input_ch, W)]
        + [
            nn.Linear(W, W)
            if i not in self.skips
            else nn.Linear(W + self.input_ch, W)
            for i in range(D - 1)
        ]
    )

    self.coeff_linear = nn.Linear(W, num_basis * 3)
    self.coeff_linear.weight.data.fill_(0.0)
    self.coeff_linear.bias.data.fill_(0.0)

  def forward(self, x):
    input_pts = self.xyzt_embed(x)

    h = input_pts
    for i, l in enumerate(self.pts_linears):
      h = self.pts_linears[i](h)
      h = F.relu(h)
      if i in self.skips:
        h = torch.cat([input_pts, h], -1)

    # sf = nn.functional.tanh(self.sf_linear(h))
    pred_coeff = self.coeff_linear(h)

    return pred_coeff / self.sf_mag_div
