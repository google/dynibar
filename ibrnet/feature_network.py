"""Class definition for 2D feature extractor."""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_for_name(module_name, class_name):
  m = importlib.import_module(module_name)
  return getattr(m, class_name)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
  """3x3 convolution with padding."""
  return nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=3,
      stride=stride,
      padding=dilation,
      groups=groups,
      bias=False,
      dilation=dilation,
      padding_mode='reflect',
  )


def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution layer."""
  return nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=1,
      stride=stride,
      bias=False,
      padding_mode='reflect',
  )


class BasicBlock(nn.Module):
  """Basic CNN block."""
  expansion = 1

  def __init__(
      self,
      inplanes,
      planes,
      stride=1,
      downsample=None,
      groups=1,
      base_width=64,
      dilation=1,
      norm_layer=None,
  ):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = nn.InstanceNorm2d

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  """Bottleneck CNN block."""

  expansion = 4

  def __init__(
      self,
      inplanes,
      planes,
      stride=1,
      downsample=None,
      groups=1,
      base_width=64,
      dilation=1,
      norm_layer=None,
  ):
    super(Bottleneck, self).__init__()
    if norm_layer is None:
      norm_layer = nn.InstanceNorm2d
    width = int(planes * (base_width / 64.0)) * groups
    self.conv1 = conv1x1(inplanes, width)
    self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
    self.conv2 = conv3x3(width, width, stride, groups, dilation)
    self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
    self.conv3 = conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(
        planes * self.expansion, track_running_stats=False, affine=True
    )
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class conv(nn.Module):
  """Convolutional layer."""

  def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
    super(conv, self).__init__()
    self.kernel_size = kernel_size
    self.conv = nn.Conv2d(
        num_in_layers,
        num_out_layers,
        kernel_size=kernel_size,
        stride=stride,
        padding=(self.kernel_size - 1) // 2,
        padding_mode='reflect',
    )
    self.bn = nn.InstanceNorm2d(
        num_out_layers, track_running_stats=False, affine=True
    )

  def forward(self, x):
    return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
  """Convolutional layers followed by upsampling."""

  def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
    super(upconv, self).__init__()
    self.scale = scale
    self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

  def forward(self, x):
    x = nn.functional.interpolate(
        x, scale_factor=self.scale, align_corners=True, mode='bilinear'
    )
    return self.conv(x)


class ResNet(nn.Module):
  """Main ResNet based feature extractor."""
  def __init__(
      self,
      encoder='resnet34',
      coarse_out_ch=32,
      fine_out_ch=32,
      norm_layer=None,
      coarse_only=False,
  ):
    super(ResNet, self).__init__()
    assert encoder in [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
    ], 'Incorrect encoder type'
    if encoder in ['resnet18', 'resnet34']:
      filters = [64, 128, 256, 512]
    else:
      filters = [256, 512, 1024, 2048]
    self.coarse_only = coarse_only
    if self.coarse_only:
      fine_out_ch = 0
    self.coarse_out_ch = coarse_out_ch
    self.fine_out_ch = fine_out_ch
    out_ch = coarse_out_ch + fine_out_ch

    # original
    layers = [3, 4, 6, 3]
    if norm_layer is None:
      # norm_layer = nn.InstanceNorm2d
      norm_layer = nn.InstanceNorm2d
    self._norm_layer = norm_layer
    self.dilation = 1
    block = BasicBlock
    replace_stride_with_dilation = [False, False, False]
    self.inplanes = 64
    self.groups = 1
    self.base_width = 64
    self.conv1 = nn.Conv2d(
        3,
        self.inplanes,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
        padding_mode='reflect',
    )
    self.bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
    self.layer2 = self._make_layer(
        block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
    )
    self.layer3 = self._make_layer(
        block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
    )

    # decoder
    self.upconv3 = upconv(filters[2], 128, 3, 2)
    self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
    self.upconv2 = upconv(128, 64, 3, 2)
    self.iconv2 = conv(filters[0] + 64, out_ch, 3, 1)

    # fine-level conv
    self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(
              planes * block.expansion, track_running_stats=False, affine=True
          ),
      )

    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            norm_layer,
        )
    )
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              groups=self.groups,
              base_width=self.base_width,
              dilation=self.dilation,
              norm_layer=norm_layer,
          )
      )

    return nn.Sequential(*layers)

  def skipconnect(self, x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(
        x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
    )

    x = torch.cat([x2, x1], dim=1)
    return x

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))

    x1 = self.layer1(x)
    x_out = self.out_conv(x1)

    x_coarse = x_out[:, : self.coarse_out_ch, :]
    x_fine = x_out[:, -self.fine_out_ch :, :]

    return x_coarse, x_fine
