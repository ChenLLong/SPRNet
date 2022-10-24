import torch
import numpy as np
import mmcv
from copy import deepcopy


def regress_point_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_points_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(regress_point_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_points_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def regress_point_single(pos_proposals, pos_assigned_gt_inds, gt_points, cfg):
    # mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    point_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_point = gt_points[pos_assigned_gt_inds[i]]
            # bbox = proposals_np[i, :].astype(np.int32)
            # x1, y1, x2, y2 = bbox
            # w = np.maximum(x2 - x1 + 1, 1)
            # h = np.maximum(y2 - y1 + 1, 1)
            # # mask is uint8 both before and after resizing
            # target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
            #                        (mask_size, mask_size))
            point_targets.append(gt_point)
        # point_targets = torch.from_numpy(np.stack(point_targets)).float().to(
        #     pos_proposals.device)
        point_targets = torch.stack(point_targets).float()
    else:
        point_targets = pos_proposals.new_zeros((0,) + tuple(gt_points.size())[1:])
    return point_targets

# def regress_point_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_center_line_list, gt_control_line_list,
#                          gt_direction_list, cfg):
#     cfg_list = [cfg for _ in range(len(pos_proposals_list))]
#     targets = map(regress_point_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_center_line_list,
#                   gt_control_line_list, gt_direction_list, cfg_list)
#     center_line_targets = []
#     control_line_angle_targets = []
#     control_line_x_targets = []
#     control_line_y_targets = []
#     control_line_length_targets = []
#     direction_targets = []
#     for target in targets:
#         center_line_targets.append(target[0])
#         control_line_angle_targets.append(target[1])
#         control_line_x_targets.append(target[2])
#         control_line_y_targets.append(target[3])
#         control_line_length_targets.append(target[4])
#         direction_targets.append(target[5])
#     center_line_targets = torch.cat(center_line_targets)
#     control_line_angle_targets = torch.cat(control_line_angle_targets)
#     control_line_x_targets = torch.cat(control_line_x_targets)
#     control_line_y_targets = torch.cat(control_line_y_targets)
#     control_line_length_targets = torch.cat(control_line_length_targets)
#     direction_targets = torch.cat(direction_targets).unsqueeze(dim=-1)
#     return center_line_targets, control_line_angle_targets, control_line_x_targets, control_line_y_targets, control_line_length_targets, direction_targets
#
#
# def regress_point_single(pos_proposals, pos_assigned_gt_inds, center_lines, control_lines, directions, cfg):
#     # mask_size = cfg.mask_size
#     num_pos = pos_proposals.size(0)
#     point_targets = []
#     center_line_targets = []
#     control_line_angle_targets = []
#     control_line_x_targets = []
#     control_line_y_targets = []
#     control_line_length_targets = []
#     direction_targets = []
#     # print(pos_proposals)
#     if num_pos > 0:
#         proposals_np = pos_proposals.cpu().numpy().astype(np.float)
#         pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
#         center_lines = center_lines.cpu().numpy().astype(np.float)
#         control_lines = control_lines.cpu().numpy().astype(np.float)
#         directions = directions.cpu().numpy().astype(np.int)
#         for i in range(num_pos):
#             bbox_proposal = proposals_np[i, :].astype(np.float)
#             proposal_mid_x = (bbox_proposal[2] + bbox_proposal[0]) / 2
#             proposal_mid_y = (bbox_proposal[3] + bbox_proposal[1]) / 2
#             proposal_height = bbox_proposal[3] - bbox_proposal[1]
#             proposal_width = bbox_proposal[2] - bbox_proposal[0]
#
#             center_points = deepcopy(center_lines[pos_assigned_gt_inds[i]])
#             line_points = deepcopy(control_lines[pos_assigned_gt_inds[i]])
#             direction = deepcopy(directions[pos_assigned_gt_inds[i]][0])
#
#             center_points[:, 0] -= proposal_mid_x
#             center_points[:, 1] -= proposal_mid_y
#             center_points[:, 0] /= proposal_width
#             center_points[:, 1] /= proposal_height
#
#             line_points[:, :, 0] -= proposal_mid_x
#             line_points[:, :, 1] -= proposal_mid_y
#             line_points[:, :, 0] /= proposal_width
#             line_points[:, :, 1] /= proposal_height
#
#             delta_x = line_points[:, 0, 0] - line_points[:, 2, 0]
#             delta_y = line_points[:, 0, 1] - line_points[:, 2, 1]
#             if direction == 0:
#                 line_angle = np.arctan(np.divide(delta_x, delta_y))
#                 if np.isnan(line_angle).any():
#                     print(control_lines)
#                     print(control_lines[pos_assigned_gt_inds[i]])
#                     print(bbox_proposal)
#                     print(line_points)
#                     print(delta_x)
#                     print(delta_y)
#                     print('line_angle{}'.format(line_angle))
#                     raise NotImplementedError
#             elif direction == 1:
#                 line_angle = np.arctan(np.divide(delta_y, delta_x))
#             else:
#                 raise NotImplementedError
#             line_x = line_points[:, 1, 0]
#             line_y = line_points[:, 1, 1]
#
#             control_line_length = np.array([line_points[:, 0, :] - line_points[:, 2, :]]) ** 2
#             control_line_length = np.squeeze(control_line_length)
#             control_line_length = np.sum(control_line_length, axis=1) ** (0.5)
#
#             center_line_targets.append(center_points)
#             control_line_angle_targets.append(line_angle)
#             control_line_x_targets.append(line_x)
#             control_line_y_targets.append(line_y)
#             control_line_length_targets.append(control_line_length)
#             direction_targets.append(direction)
#
#         center_line_targets = torch.from_numpy(np.stack(center_line_targets)).float().to(pos_proposals.device)
#         control_line_angle_targets = torch.from_numpy(np.stack(control_line_angle_targets)).float().to(
#             pos_proposals.device)
#         control_line_x_targets = torch.from_numpy(np.stack(control_line_x_targets)).float().to(pos_proposals.device)
#         control_line_y_targets = torch.from_numpy(np.stack(control_line_y_targets)).float().to(pos_proposals.device)
#         control_line_length_targets = torch.from_numpy(np.stack(control_line_length_targets)).float().to(
#             pos_proposals.device)
#         direction_targets = torch.from_numpy(np.stack(direction_targets)).float().to(pos_proposals.device)
#         # print(center_line_targets,control_line_angle_targets,control_line_x_targets,control_line_y_targets,control_line_length_targets,direction_targets)
#     else:
#         point_targets = pos_proposals.new_zeros((0, m))
#     return center_line_targets, control_line_angle_targets, control_line_x_targets, control_line_y_targets, control_line_length_targets, direction_targets
