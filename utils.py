"""Utility functions."""

import cv2
import matplotlib as mpl
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10.0 * np.log(x + TINY_NUMBER) / np.log(10.0)


def img2mse(x, y, mask=None):
  """MSE between two images."""
  if mask is None:
    return torch.mean((x - y) * (x - y))
  else:
    return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (
        torch.sum(mask) * x.shape[-1] + TINY_NUMBER
    )


def img2charbonier(x, y, mask=None, eps=0.001):
  """Charbonier loss between two images."""
  if mask is None:
    return torch.mean(torch.sqrt((x - y) ** 2 + eps**2))
  else:
    return torch.sum(
        torch.sqrt((x - y) ** 2 + eps**2) * mask.unsqueeze(-1)
    ) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def img2psnr(x, y, mask=None):
  return mse2psnr(img2mse(x, y, mask).item())


def cycle(iterable):
  while True:
    for x in iterable:
      yield x


def get_vertical_colorbar(
    h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2
):
  """Get colorbar."""
  fig = Figure(figsize=(2, 8), dpi=100)
  fig.subplots_adjust(right=1.5)
  canvas = FigureCanvasAgg(fig)

  # Do some plotting.
  ax = fig.add_subplot(111)
  cmap = cm.get_cmap(cmap_name)
  norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

  tick_cnt = 6
  tick_loc = np.linspace(vmin, vmax, tick_cnt)
  cb1 = mpl.colorbar.ColorbarBase(
      ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation='vertical'
  )

  tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
  if cbar_precision == 0:
    tick_label = [x[:-2] for x in tick_label]

  cb1.set_ticklabels(tick_label)

  cb1.ax.tick_params(labelsize=18, rotation=0)

  if label is not None:
    cb1.set_label(label)

  fig.tight_layout()

  canvas.draw()
  s, (width, height) = canvas.print_to_buffer()

  im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

  im = im[:, :, :3].astype(np.float32) / 255.0
  if h != im.shape[0]:
    w = int(im.shape[1] / im.shape[0] * h)
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

  return im


def colorize_np(
    x,
    cmap_name='jet',
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
    cbar_precision=2,
):
  """turn a grayscale image into a color image."""
  if range is not None:
    vmin, vmax = range
  elif mask is not None:
    # vmin, vmax = np.percentile(x[mask], (2, 100))
    vmin = np.min(x[mask][np.nonzero(x[mask])])
    vmax = np.max(x[mask])
    # vmin = vmin - np.abs(vmin) * 0.01
    x[np.logical_not(mask)] = vmin
    # print(vmin, vmax)
  else:
    vmin, vmax = np.percentile(x, (1, 99))
    vmax += TINY_NUMBER

  x = np.clip(x, vmin, vmax)
  x = (x - vmin) / (vmax - vmin)
  x = np.clip(x, 0.0, 1.0)

  cmap = cm.get_cmap(cmap_name)
  x_new = cmap(x)[:, :, :3]

  if mask is not None:
    mask = np.float32(mask[:, :, np.newaxis])
    x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

  cbar = get_vertical_colorbar(
      h=x.shape[0],
      vmin=vmin,
      vmax=vmax,
      cmap_name=cmap_name,
      cbar_precision=cbar_precision,
  )

  if append_cbar:
    if cbar_in_image:
      x_new[:, -cbar.shape[1] :, :] = cbar
    else:
      x_new = np.concatenate(
          (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
      )
    return x_new
  else:
    return x_new


# tensor
def colorize(
    x,
    cmap_name='jet',
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
):
  """Convert gray scale image such as depth to RGB image."""
  device = x.device
  x = x.cpu().numpy()
  if mask is not None:
    mask = mask.cpu().numpy() > 0.99
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

  x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
  x = torch.from_numpy(x).to(device)
  return x
