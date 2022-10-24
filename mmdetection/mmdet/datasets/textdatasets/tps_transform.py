import numpy as np
import copy
import torch
import itertools
import torch.nn as nn
from torch.autograd import Function
import random

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

def tps_transform(source_height, source_width, target_height, target_width, bboxes, length):
    source_control_points = torch.Tensor(list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / 20),
        torch.arange(-1.0, 1.00001, 2.0),
    )))

    source_control_points = torch.cat((source_control_points[0::2, :], source_control_points[1::2, :]), dim=0)

    bash_anchor = torch.arange(-1.0, 1.00001, 2 / 20)
    # 0.2*x^2-0.2*x,0.2*x^2+0.2*x,-0.2*x^2+0.2*x,-0.2*x^2-0.2*x,0.2*x^2,-0.2*x^2,
    # 0.2*x^3-0.4*x,0.3*x^3+0.3*x^2-0.3*x

    tps_funcs = [
        [0, 0.3, -0.2, 0],
        [0, 0.2, 0.2, 0],
        [0, -0.2, 0.2, 0],
        [0, -0.2, -0.2, 0],
        [0, -0.4, 0, 0],
        [0.2, 0, -0.4, 0],
        [0.5, 0.3, -0.3, 0],
        [-0.5, -0.3, 0.3, 0],
        [0.5, 0.1, -0.3, 0],
        [0.5, 0.1, -0.4, 0],
        [0.5, 0.1, -0.5, 0],
        [-0.5, -0.1, 0.3, 0],
        [1.0, 0.1, -0.9, 0],
        [-1.0, -0.1, 0.9, 0]
    ]

    random_ind=random.randint(0,len(tps_funcs)-1)
    poly_coeff = tps_funcs[random_ind]
    anchor_value = poly_coeff[0] * bash_anchor ** 3 + poly_coeff[1] * bash_anchor ** 2 + poly_coeff[2] * bash_anchor ** 1 + poly_coeff[3]

    max_value = torch.max(anchor_value)
    up_delta = 1.0 - max_value

    min_value = torch.min(anchor_value)
    down_delta = -1.0 - min_value

    points = torch.stack((bash_anchor, anchor_value), dim=-1)

    down_points = copy.deepcopy(points)
    down_points[:, 1] = down_points[:, 1] + down_delta

    up_points = copy.deepcopy(points)
    up_points[:, 1] = up_points[:, 1] + up_delta

    target_control_points = torch.cat((down_points, up_points), dim=0)

    # assert target_control_points.ndimension() == 2
    # assert target_control_points.size(1) == 2
    N = target_control_points.size(0)
    target_control_points = target_control_points.float()

    # create padded kernel matrix
    forward_kernel = torch.zeros(N + 3, N + 3)
    target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
    forward_kernel[:N, :N].copy_(target_control_partial_repr)
    forward_kernel[:N, -3].fill_(1)
    forward_kernel[-3, :N].fill_(1)
    forward_kernel[:N, -2:].copy_(target_control_points)
    forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
    # compute inverse matrix
    inverse_kernel = torch.inverse(forward_kernel)

    # create target cordinate matrix
    HW = target_height * target_width
    target_coordinate = list(itertools.product(range(target_height), range(target_width)))
    target_coordinate = torch.Tensor(target_coordinate) # HW x 2
    Y, X = target_coordinate.split(1, dim=1)
    Y = Y * 2 / (target_height - 1) - 1
    X = X * 2 / (target_width - 1) - 1
    target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
    target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
    target_coordinate_repr = torch.cat([
        target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
    ], dim=1)

    padding_matrix = torch.zeros(3, 2)
    source_control_points=torch.unsqueeze(source_control_points, dim=0)
    # assert source_control_points.ndimension() == 3
    # assert source_control_points.size(1) == num_points
    # assert source_control_points.size(2) == 2
    batch_size = source_control_points.size(0)

    Y = torch.cat([source_control_points, padding_matrix.expand(batch_size, 3, 2)], 1)
    mapping_matrix = torch.matmul(inverse_kernel, Y)
    source_coordinate_map = torch.matmul(target_coordinate_repr, mapping_matrix)

    box_points = torch.from_numpy(np.concatenate(bboxes, axis=0)).float()
    box_points[:, 0] = box_points[:, 0] * 2 / (source_width - 1) - 1
    box_points[:, 1] = box_points[:, 1] * 2 / (source_height - 1) - 1
    try:
        source_coordinate = source_coordinate_map.repeat(box_points.size(0), 1, 1)
        distance = source_coordinate.permute(0, 2, 1) - box_points.unsqueeze(dim=-1)
        distance = torch.abs(distance).permute(0, 2, 1).sum(dim=-1, keepdim=False)
        min_distance_ind = torch.argmin(distance, dim=-1, keepdim=False)
    except RuntimeError:
        min_distance_inds = []
        source_coordinate = source_coordinate_map.squeeze(dim=0)
        for i in range(box_points.size(0)):
            distance = source_coordinate.permute(1, 0) - box_points[i].unsqueeze(dim=-1)
            distance = torch.abs(distance).permute(1, 0).sum(dim=-1, keepdim=False)
            min_distance_ind = torch.argmin(distance, dim=0, keepdim=False)
            min_distance_inds.append(min_distance_ind)
        min_distance_ind = torch.stack(min_distance_inds, dim=0)
    target_box = target_coordinate[min_distance_ind]

    target_box[:, 0] = (target_box[:, 0] + 1) * (target_width - 1) / 2
    target_box[:, 1] = (target_box[:, 1] + 1) * (target_height - 1) / 2

    target_box = target_box.numpy().astype(np.int)
    split_cumsum = np.cumsum(length[:-1])
    target_box = np.vsplit(target_box, split_cumsum)

    return source_coordinate_map, target_box