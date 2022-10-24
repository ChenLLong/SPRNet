import torch
import numpy as np
import mmcv
from copy import deepcopy


def recognition_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_recognition_list,
                       cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(recognition_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_recognition_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def recognition_single(pos_proposals, pos_assigned_gt_inds, gt_points, cfg):
    # mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    point_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_point = gt_points[pos_assigned_gt_inds[i]]
            point_targets.append(gt_point)
        point_targets = torch.stack(point_targets).long()
    else:
        point_targets = pos_proposals.new_zeros((0, ) + tuple(gt_points.size())[1:])
    return point_targets
