# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch
from utils import img2charbonier

EPSILON = 0.001

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, motion_mask=None):
        '''
        training criterion
        '''
        pred_rgb = outputs['rgb']
        pred_mask = outputs['mask'].float()
        gt_rgb = ray_batch['rgb']

        if motion_mask is not None:
            pred_mask = pred_mask * motion_mask.float()

        loss = img2charbonier(pred_rgb, gt_rgb, pred_mask, EPSILON)

        return loss



def compute_temporal_rgb_loss(outputs, ray_batch, motion_mask=None):
    pred_rgb = outputs['rgb']
    gt_rgb = ray_batch['rgb']

    occ_weight_map = outputs['occ_weight_map']
    pred_mask = outputs['mask'].float()

    if motion_mask is not None:
        pred_mask = pred_mask * motion_mask

    final_w = pred_mask * occ_weight_map
    final_w = final_w.unsqueeze(-1).repeat(1, 3)

    loss = torch.sum(final_w * torch.sqrt((pred_rgb - gt_rgb)**2 + EPSILON**2) ) / (torch.sum(final_w) + 1e-8)
    return loss

def compute_rgb_loss(pred_rgb, ray_batch, pred_mask):
    gt_rgb = ray_batch['rgb']
    loss = img2charbonier(pred_rgb, gt_rgb, pred_mask, EPSILON)

    return loss

# def compute_mask_ssi_depth_loss(pred_depth, gt_depth, mask): 
#     t_pred = torch.median(pred_depth)
#     s_pred = torch.mean(torch.abs(pred_depth - t_pred))

#     t_gt = torch.median(gt_depth)
#     s_gt = torch.mean(torch.abs(gt_depth - t_gt))

#     pred_depth_n = (pred_depth - t_pred) / s_pred
#     gt_depth_n = (gt_depth - t_gt) / s_gt

#     num_pixel = torch.sum(mask) + 1e-8

#     return torch.sum(torch.abs(pred_depth_n - gt_depth_n) * mask)/num_pixel


def compute_entropy(x):
    return -torch.mean(x * torch.log(x + 1e-8)) 


def compute_flow_loss(render_flow, gt_flow, gt_mask):
    gt_mask_rep = gt_mask.repeat(1, 1, 2)
    return torch.sum(torch.abs(render_flow - gt_flow) * gt_mask_rep) / (torch.sum(gt_mask_rep) + 1e-8)
