import os.path as osp
import warnings

import mmcv
import imageio
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES
from .polybbox_v1 import PolyBBox as PolyBBox_divide

import copy
import torch
import itertools
import torch.nn as nn
from torch.autograd import Function
import random


import cv2
import numpy as np
import copy
EPSILON = 1e-8


import torch.nn.functional as F
from torch.autograd import Variable


@PIPELINES.register_module
class LoadTextAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            if results['img_prefix'] is not None:
                file_path = osp.join(results['img_prefix'],
                                     results['img_info']['filename'])
            else:
                file_path = results['img_info']['filename']
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_poly(self, results):
        for key in ['center_line', 'ori_poly', 'width_line', 'line_angle', 'line_length', 'line_x', 'line_y', 'direction', 'poly_pts', 'transcriptions']:
            results['gt_'+key] = results['ann_info'][key]
            results['poly_fields'].append('gt_'+key)
        results['rotate_angle'] = 0
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
            results = self._load_poly(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str

@PIPELINES.register_module
class LoadTextImageFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        # img = mmcv.imread(filename)
        img = self.read_image(filename)
        if self.to_float32:
            img = img.astype(np.float32)

        if np.random.rand() < 0.:
            source_height, source_width, _ = img.shape
            target_height = source_height
            target_width = source_width
            bboxes = []
            length = []
            for i, poly_item in enumerate(results['ann_info']['ori_poly']):
                # bbox=poly, script=script, transcription=transcription
                # poly = np.array([int(x) for x in poly]).reshape(-1, 2)
                poly_split = PolyBBox(poly_item, num_width_line=11).gen_poly_label()
                if poly_split is None:
                    poly_split = poly_item
                length.append(poly_split.shape[0])
                bboxes.append(poly_split)
            for i, poly_item_ignore in enumerate(results['ann_info']['bboxes_ignore']):
                poly_split = PolyBBox(poly_item_ignore, num_width_line=11).gen_poly_label()
                if poly_split is None:
                    poly_split = poly_item
                length.append(poly_split.shape[0])
                bboxes.append(poly_split)
            if bboxes:
                with torch.no_grad():
                    source_coordinate, target_box = tps_transform(source_height, source_width, target_height, target_width, bboxes, length)
                    grid = source_coordinate.view(1, target_height, target_width, 2)
                    # canvas = Variable(torch.Tensor(1, 3, target_height, target_width).fill_(0))
                    canvas = None
                    img = np.expand_dims(img.swapaxes(2, 1).swapaxes(1, 0), 0)
                    img = torch.from_numpy(img.astype(np.float32))
                    img = grid_sample(img, grid, canvas)
                    img = img.data.cpu().numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2)

                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_ori_poly = []
                gt_masks_ann = []
                gt_center_line = []
                gt_width_line = []
                gt_line_angle = []
                gt_line_length = []
                gt_line_x = []
                gt_line_y = []
                gt_direction = []
                gt_poly_pts = []
                gt_transcriptions = []
                for i, poly in enumerate(target_box[:len(results['ann_info']['ori_poly'])]):
                    poly_label = PolyBBox_divide(poly, num_width_line=9).gen_poly_label()

                    if poly_label is None:
                        bbox = (np.min(poly[:, 0]), np.min(poly[:, 1]),
                                np.max(poly[:, 0]), np.max(poly[:, 1]))
                        gt_bboxes_ignore.append(bbox)
                    # elif script == '#':
                    #     gt_bboxes_ignore.append(poly_label['bbox'])
                    else:
                        gt_bboxes.append(poly_label['bbox'])
                        gt_labels.append(results['ann_info']['labels'][i])
                        gt_ori_poly.append(poly_label['poly'])
                        gt_masks_ann.append([poly_label['poly'].flatten().tolist()])
                        gt_center_line.append(poly_label['center_line'])
                        gt_width_line.append(poly_label['width_line'])
                        gt_line_angle.append(poly_label['line_angle'])
                        gt_line_length.append(poly_label['line_length'])
                        gt_line_x.append(poly_label['line_x'])
                        gt_line_y.append(poly_label['line_y'])
                        gt_direction.append(poly_label['direction'])
                        gt_poly_pts.append(poly_label['origin_poly_pts'])
                        gt_transcriptions.append(results['ann_info']['transcriptions'][i])

                for i, poly_ignore in enumerate(target_box[len(results['ann_info']['ori_poly']):]):
                    bbox = (np.min(poly_ignore[:, 0]), np.min(poly_ignore[:, 1]),
                            np.max(poly_ignore[:, 0]), np.max(poly_ignore[:, 1]))
                    gt_bboxes_ignore.append(bbox)

                if gt_bboxes:
                    gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                    gt_labels = np.array(gt_labels, dtype=np.int64)
                    # gt_masks_ann = gt_masks_ann
                else:
                    gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                    gt_labels = np.array([], dtype=np.int64)
                    gt_masks_ann = [[[0, 0, 0, 0, 0, 0]]]

                if gt_bboxes_ignore:
                    gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
                else:
                    gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
                ann = dict(
                    bboxes=gt_bboxes,
                    labels=gt_labels,
                    bboxes_ignore=gt_bboxes_ignore,
                    ori_poly=gt_ori_poly,
                    masks=gt_masks_ann,
                    center_line=gt_center_line,
                    width_line=gt_width_line,
                    line_angle=gt_line_angle,
                    line_length=gt_line_length,
                    line_x=gt_line_x,
                    line_y=gt_line_y,
                    direction=gt_direction,
                    poly_pts=gt_poly_pts,
                    transcriptions=gt_transcriptions)
                results['ann_info'] = ann
        else:
            gt_bboxes_ignore=[]
            for poly in results['ann_info']['bboxes_ignore']:
                gt_bboxes_ignore.append((np.min(poly[:, 0]), np.min(poly[:, 1]),
                                         np.max(poly[:, 0]), np.max(poly[:, 1])))
            if gt_bboxes_ignore:
                results['ann_info']['bboxes_ignore'] = np.array(gt_bboxes_ignore, dtype=np.float32)
            else:
                results['ann_info']['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)

        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # results['img_info']['height'], results['img_info']['width'],c=img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)

    def read_image(self, im_fp):
        if im_fp.endswith('.gif'):
            im = imageio.mimread(im_fp)
            if im is not None:
                im = np.array(im)[0][:, :, 0:3]
        else:
            im = mmcv.imread(im_fp)

        return im

def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

def norm2(x, axis=None):
    return np.sqrt(np.sum(x ** 2, axis=axis))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2) + EPSILON)


class PolyBBox(object):

    def __init__(self, pts, n_parts=16, num_width_line=5):
        # self.pts = self.remove_points(pts)
        self.pts = pts
        self.pts_num = len(self.pts)
        self.n_parts = n_parts
        self.num_width_line = num_width_line

        self.bbox = (np.min(self.pts[:, 0]), np.min(self.pts[:, 1]),
                     np.max(self.pts[:, 0]), np.max(self.pts[:, 1]))
        self.cx, self.cy = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
        self.height, self.width = (self.bbox[3] - self.bbox[1]), (self.bbox[2] - self.bbox[0])

    def remove_points(self, pts):
        '''
        remove point if area is almost unchanged after removing it
        '''
        rm_pts_idx = []
        if len(pts) > 4:
            ori_area = cv2.contourArea(pts)
            for i in range(len(pts)):
                # attempt to remove pts[i]
                index = list(range(len(pts)))
                index.remove(i)
                area = cv2.contourArea(pts[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(pts) - len(rm_pts_idx) > 4:
                    rm_pts_idx.append(i)
        return np.array([pt for i, pt in enumerate(pts) if i not in rm_pts_idx])

    def find_short_sides(self):
        if self.pts_num > 4:
            points = np.concatenate([self.pts, self.pts[:3]])
            candidate = []
            for i in range(1, self.pts_num + 1):
                prev_edge = points[i] - points[i - 1]
                next_edge = points[i + 2] - points[i + 1]
                if cos(prev_edge, next_edge) < -0.8:
                    candidate.append((i % self.pts_num, (i + 1) % self.pts_num, norm2(points[i] - points[i + 1])))

            if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
                # if candidate number < 2, or two bottom are joined, select 2 farthest edge
                mid_list = []
                for i in range(self.pts_num):
                    mid_point = (points[i] + points[(i + 1) % self.pts_num]) / 2
                    mid_list.append((i, (i + 1) % self.pts_num, mid_point))

                dist_list = []
                for i in range(self.pts_num):
                    for j in range(self.pts_num):
                        s1, e1, mid1 = mid_list[i]
                        s2, e2, mid2 = mid_list[j]
                        dist = norm2(mid1 - mid2)
                        dist_list.append((s1, e1, s2, e2, dist))
                short_sides_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
                short_sides = [dist_list[short_sides_idx[0]][:2], dist_list[short_sides_idx[1]][:2]]
                if short_sides[0][0] == short_sides[1][1] or short_sides[0][1] == short_sides[1][0]:
                    short_sides_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-1:]
                    short_sides = [dist_list[short_sides_idx[0]][0:2], dist_list[short_sides_idx[0]][2:4]]
            else:
                short_sides = [candidate[0][:2], candidate[1][:2]]

        else:
            d1 = norm2(self.pts[1] - self.pts[0]) + norm2(self.pts[2] - self.pts[3])
            d2 = norm2(self.pts[2] - self.pts[1]) + norm2(self.pts[0] - self.pts[3])
            short_sides = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
        assert len(short_sides) == 2, 'more or less than 2 short sides'
        return short_sides

    def find_long_sides(self, short_sides):
        ss1_start, ss1_end = short_sides[0]
        ss2_start, ss2_end = short_sides[1]

        long_sides_list = [[], []]
        i = (ss1_end + 1) % self.pts_num
        while (i != ss2_end):
            long_sides_list[0].append(((i - 1) % self.pts_num, i))
            i = (i + 1) % self.pts_num

        i = (ss2_end + 1) % self.pts_num
        while (i != ss1_end):
            long_sides_list[1].append(((i - 1) % self.pts_num, i))
            i = (i + 1) % self.pts_num

        return long_sides_list

    def partition_long_sides(self):
        '''
        cover text region with several parts
        :return:
        '''

        self.short_sides = self.find_short_sides()  # find two short sides of this Text
        self.long_sides_list = self.find_long_sides(self.short_sides)  # find two long sides sequence

        inner_pts1 = self.split_side_seqence(self.long_sides_list[0])
        inner_pts2 = self.split_side_seqence(self.long_sides_list[1])
        inner_pts2 = inner_pts2[::-1]  # innverse one of long edge

        center_pts = (inner_pts1 + inner_pts2) / 2  # disk center

        return inner_pts1, inner_pts2, center_pts  # , radii

    def split_side_seqence(self, long_sides):
        side_lengths = [norm2(self.pts[v1] - self.pts[v2]) for v1, v2 in long_sides]
        sides_cumsum = np.cumsum([0] + side_lengths)
        total_length = sum(side_lengths)
        length_per_part = total_length / self.n_parts

        cur_node = 0  # first point
        mid_pt_list = []

        for i in range(1, self.n_parts):
            curr_len = i * length_per_part

            while (curr_len > sides_cumsum[cur_node + 1]):
                cur_node += 1

            v1, v2 = long_sides[cur_node]
            pt1, pt2 = self.pts[v1], self.pts[v2]

            # start_point = self.pts[long_edge[cur_node]]
            end_shift = curr_len - sides_cumsum[cur_node]
            ratio = end_shift / side_lengths[cur_node]
            new_pt = pt1 + ratio * (pt2 - pt1)

            mid_pt_list.append(new_pt)

        # add first and last point
        pt_first = self.pts[long_sides[0][0]]
        pt_last = self.pts[long_sides[-1][1]]
        mid_pt_list = [pt_first] + mid_pt_list + [pt_last]
        return np.stack(mid_pt_list)

    def check_and_validate_polys(self):
        len_pts = len(self.pts)
        for i in range(len_pts-2):
            for j in range(i+2, len_pts):
                if (j+1) % len_pts !=i:
                    if self.isIntersect(self.pts[i], self.pts[i+1], self.pts[j], self.pts[(j+1) % len_pts]):
                        return False
        return True

    def isIntersect(self, p1, p2, q1, q2):
        a1 = (p2[0] - p1[0]) * (q1[1] - p1[1]) - (q1[0] - p1[0]) * (p2[1] - p1[1])
        a2 = (p2[0] - p1[0]) * (q2[1] - p1[1]) - (q2[0] - p1[0]) * (p2[1] - p1[1])
        b1 = (q2[0] - q1[0]) * (p1[1] - q1[1]) - (p1[0] - q1[0]) * (q2[1] - q1[1])
        b2 = (q2[0] - q1[0]) * (p2[1] - q1[1]) - (p2[0] - q1[0]) * (q2[1] - q1[1])

        if a1 * a2 < 0 and b1 * b2 < 0:
            return True
        return False

    def gen_poly_label(self):
        if self.pts_num < 4:
            # print('WARNING: number of points is {} less than 4.\n'.format(self.pts_num))
            # print(self.pts)
            return None

        if not self.check_and_validate_polys():
            # print('WARNING: illegal polygon {}.\n'.format(self.pts))
            return None

        self.short_sides = self.find_short_sides()  # find two short sides of this Text
        self.long_sides_list = self.find_long_sides(self.short_sides)  # find two long sides sequence

        if len(self.long_sides_list[0]) == 0 or len(self.long_sides_list[1]) == 0 or len(self.long_sides_list[0]) != len(self.long_sides_list[1]):
            # print('WARNING: short and long sides {}\t{}.'.format(self.short_sides, self.long_sides_list))
            return None

        border_pts1 =[self.pts[v1] for v1, _ in self.long_sides_list[0]]
        border_pts1 += [self.pts[self.long_sides_list[0][-1][1]]]
        border_pts2 = [self.pts[v1] for v1, _ in self.long_sides_list[1]]
        border_pts2 += [self.pts[self.long_sides_list[1][-1][1]]]
        border_pts2 = border_pts2[::-1]

        side_pts1 = self.split_side_seqence(self.long_sides_list[0])
        side_pts2 = self.split_side_seqence(self.long_sides_list[1])
        # side_pts2 = side_pts2[::-1]  # innverse one of long edge

        return np.concatenate((side_pts1, side_pts2), axis=0)




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


def tps_transform(source_height, source_width, target_height, target_width, bboxes, length, num_anchor_point = 20):
    bash_anchor = torch.linspace(-1, 1, target_width)

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

    random_coff = random.random() -0.5
    random_ind=random.randint(0,len(tps_funcs)-1)
    poly_coeff = tps_funcs[random_ind]
    bash_anchor_1 = bash_anchor[0:target_width//2+1]
    anchor_value_1 = random_coff * bash_anchor_1 ** 4 + poly_coeff[0] * bash_anchor_1 ** 3 + poly_coeff[1] * bash_anchor_1 ** 2 + poly_coeff[2] * bash_anchor_1 ** 1 + poly_coeff[3]
    random_d = random.random() * np.pi
    anchor_value_1 = 0.2 * torch.sin(bash_anchor_1 * 9 + random_d)

    random_ind=random.randint(0,len(tps_funcs)-1)
    poly_coeff = tps_funcs[random_ind]
    bash_anchor_2 = bash_anchor[target_width//2:]
    anchor_value_2 = random_coff * bash_anchor_2 ** 4 + poly_coeff[0] * bash_anchor_2 ** 3 + poly_coeff[1] * bash_anchor_2 ** 2 + poly_coeff[2] * bash_anchor_2 ** 1 + poly_coeff[3]
    random_d = random.random() * np.pi
    anchor_value_2 = 0.2 * torch.cos(bash_anchor_2 * 9 + random_d)

    anchor_value_2 = anchor_value_2 - (anchor_value_2[0]-anchor_value_1[-1])
    anchor_value = torch.cat((anchor_value_1,anchor_value_2[1:]),dim=-1)

    max_value = torch.max(anchor_value)
    up_delta = - 1.0 - max_value

    min_value = torch.min(anchor_value)
    down_delta = -1.0 - min_value
    up_down_delta = max_value - min_value
    anchor_value = anchor_value + up_delta

    grid = torch.ones(1, target_height, target_width, 2, device='cpu')
    grid[:, :, :, 0] = torch.linspace(-1, 1, target_width)
    grid[:, :, :, 1] = anchor_value[None, ...].repeat(target_height, 1) + torch.linspace(0, 2+up_down_delta, target_height)[..., None].repeat(1, target_width)

    box_points = torch.from_numpy(np.concatenate(bboxes, axis=0)).float()
    box_points = torch.clamp(box_points, 0, source_width - 1)
    box_points_ind = box_points[:, 0].long()
    # box_points[:, 0] = box_points[:, 0] * 2 / (source_width - 1) - 1
    box_points[:, 1] = box_points[:, 1] * 2 / (source_height - 1) - 1
    box_points[:, 1] = (box_points[:, 1] - anchor_value[box_points_ind]) / (2+up_down_delta) * 2 -1
    box_points[:, 1] = (box_points[:, 1] + 1) * (target_height - 1) / 2

    target_box = box_points.numpy().astype(np.int)
    split_cumsum = np.cumsum(length[:-1])
    target_box = np.vsplit(target_box, split_cumsum)
    return grid, target_box

