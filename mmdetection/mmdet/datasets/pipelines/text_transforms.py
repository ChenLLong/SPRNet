import inspect

import albumentations
import mmcv
import cv2
import numpy as np
from albumentations import Compose
from imagecorruptions import corrupt
from shapely.geometry import box, Polygon
from shapely import affinity
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES


@PIPELINES.register_module
class TextResize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = masks
    def _resize_poly_pts(self,results):
        for key in ['gt_poly_pts', 'gt_center_line', 'gt_width_line', 'gt_ori_poly']:
            results[key] = [
                poly * results['scale_factor']
                for poly in results[key]]

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_poly_pts(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class TextRandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def filp_center_line(self, center_line, w, direction):
        center_line[:, 0] = w - center_line[:, 0]
        if direction == 0:
            return center_line[::-1]
        return center_line

    def filp_ori_poly(self, ori_poly, w, direction):
        ori_poly[:, 0] = w - ori_poly[:, 0]
        return ori_poly[::-1]

    def filp_line_angle(self, line_angle, direction):
        line_angle = np.negative(line_angle)
        if direction == 0:
            line_angle = line_angle[::-1]
        return line_angle

    def filp_line_xy(self, line_x, line_y, direction):
        line_x = np.negative(line_x)
        if direction == 0:
            line_x = line_x[::-1]
            line_y = line_y[::-1]
        return line_x, line_y

    def flip_poly_pts(self, poly_pts, w, direction):
        poly_pts[:,:,0] = w - poly_pts[:,:,0]
        if direction == 0:
            return poly_pts[::-1]
        else:
            return poly_pts[:,::-1,:]

    def regress_point_flip(self, gt_center_line, gt_ori_poly, gt_width_line, gt_line_angle, gt_line_x, gt_line_y,
                           gt_line_length, gt_direction, gt_poly_pts, img_shape, gt_transcriptions, **kwargs):
        w = img_shape[1]
        gt_center_line = [
            self.filp_center_line(center_line, w, gt_direction[i][0])
            for i, center_line in enumerate(gt_center_line)
        ]

        gt_ori_poly = [
            self.filp_ori_poly(ori_poly, w, gt_direction[i][0])
            for i, ori_poly in enumerate(gt_ori_poly)
        ]

        gt_width_line = [
            self.flip_poly_pts(width_line, w, gt_direction[i][0])
            for i, width_line in enumerate(gt_width_line)
        ]

        gt_line_angle = [
            self.filp_line_angle(lines_angle, gt_direction[i][0])
            for i, lines_angle in enumerate(gt_line_angle)
        ]

        xs = []
        ys = []
        for line_x, line_y, direction in zip(gt_line_x, gt_line_y, gt_direction):
            line_x, line_y = self.filp_line_xy(line_x, line_y, direction[0])
            xs.append(line_x)
            ys.append(line_y)
        gt_line_x = xs
        gt_line_y = ys

        gt_line_length = [
            lines_length[::-1] if gt_direction[i][0] == 0 else lines_length
            for i, lines_length in enumerate(gt_line_length)
        ]

        gt_poly_pts = [
            self.flip_poly_pts(poly_pt, w, gt_direction[i][0])
            for i, poly_pt in enumerate(gt_poly_pts)
        ]

        gt_transcriptions = [transcription[::-1] for transcription in gt_transcriptions]
        return dict(gt_center_line=gt_center_line,
                    gt_ori_poly=gt_ori_poly,
                    gt_width_line=gt_width_line,
                    gt_line_angle=gt_line_angle,
                    gt_line_x=gt_line_x,
                    gt_line_y=gt_line_y,
                    gt_line_length=gt_line_length,
                    gt_direction=gt_direction,
                    gt_poly_pts=gt_poly_pts,
                    gt_transcriptions=gt_transcriptions)

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
            # flip poly_label
            results.update(self.regress_point_flip(**results))
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)

@PIPELINES.register_module
class TextMinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size=640, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.2):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.augment = RandomCrop(crop_size=crop_size, max_tries=50, min_crop_side_ratio=min_crop_size)

    def __call__(self, results):

        return self.augment(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str

def regular_resize(results, crop_size):
    image = results['img']
    h, w, c = image.shape
    if h < w:
        scale_ratio = crop_size * 1.0 / w
        new_h = int(round(crop_size * h * 1.0 / w))
        if new_h > crop_size:
            new_h = crop_size
        image = mmcv.imresize(image, (crop_size, new_h))
        new_img = np.zeros((crop_size, crop_size, 3))
        new_img[:new_h, :, :] = image
        results['img'] = new_img
        if 'gt_masks' in results:
            gt_mask=[]
            for mask in results['gt_masks']:
                mask = mmcv.imresize(mask, (crop_size, new_h))
                new_mask = np.zeros((crop_size, crop_size))
                new_mask[:new_h, :] = mask
                gt_mask.append(new_mask)
            results['gt_masks'] = gt_mask
    else:
        scale_ratio = crop_size * 1.0 / h
        new_w = int(round(crop_size * w * 1.0 / h))
        if new_w > crop_size:
            new_w = crop_size
        image = mmcv.imresize(image, (new_w, crop_size))
        new_img = np.zeros((crop_size, crop_size, 3))
        new_img[:, :new_w, :] = image
        results['img'] = new_img
        if 'gt_masks' in results:
            gt_mask = []
            for mask in results['gt_masks']:
                mask = mmcv.imresize(mask, (new_w, crop_size))
                new_mask = np.zeros((crop_size, crop_size))
                new_mask[:, :new_w] = mask
                gt_mask.append(new_mask)
            results['gt_masks'] = gt_mask

    keys = ['gt_bboxes', 'gt_bboxes_ignore', 'gt_ori_poly', 'gt_poly_pts', 'gt_center_line', 'gt_width_line']
    for key in keys:
        results[key] = np.array(results[key]) * scale_ratio
    results['img_shape'] = results['img'].shape
    return results


def random_crop(results, crop_size, max_tries, w_axis, h_axis, min_crop_side_ratio):
    image = results['img']
    boxes = results['gt_bboxes']
    boxes_ignore = results['gt_bboxes_ignore']

    h, w, c = image.shape
    selected_boxes = []
    selected_boxes_ignore = []
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy)
        ymax = np.max(yy)
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        if boxes.shape[0] != 0:
            box_axis_in_area = (boxes[:, 0::2] >= xmin) & (boxes[:, 0::2] <= xmax) \
                               & (boxes[:, 1::2] >= ymin) & (boxes[:, 1::2] <= ymax)
            selected_boxes = np.where(
                np.sum(box_axis_in_area, axis=1) == 2)[0]
            if len(selected_boxes) > 0:
                if boxes_ignore.shape[0] != 0:
                    boxes_ignore_in_area = (boxes_ignore[:, 0::2] >= xmin) & (boxes_ignore[:, 0::2] <= xmax) \
                                           & (boxes_ignore[:, 1::2] >= ymin) & (boxes_ignore[:, 1::2] <= ymax)
                    selected_boxes_ignore = np.where(
                        np.sum(boxes_ignore_in_area, axis=1) == 2)[0]
                else:
                    selected_boxes_ignore = []
                break
        else:
            selected_boxes = []
            if boxes_ignore.shape[0] != 0:
                boxes_ignore_in_area = (boxes_ignore[:, 0::2] >= xmin) & (boxes_ignore[:, 0::2] <= xmax) \
                                       & (boxes_ignore[:, 1::2] >= ymin) & (boxes_ignore[:, 1::2] <= ymax)
                selected_boxes_ignore = np.where(
                    np.sum(boxes_ignore_in_area, axis=1) == 2)[0]
            else:
                selected_boxes_ignore = []
            break
    if i == max_tries - 1:
        return regular_resize(results, crop_size)

    results['gt_bboxes_ignore'] = results['gt_bboxes_ignore'][selected_boxes_ignore]

    keys = ['gt_bboxes', 'gt_labels', 'gt_poly_pts', 'gt_ori_poly', 'gt_center_line', 'gt_width_line', 'gt_line_angle', 'gt_line_length', 'gt_line_x', 'gt_line_y', 'gt_direction', 'gt_transcriptions']
    for key in keys:
        # if key == 'gt_labes':
        #     print('gt_labels {}'.format(results[key].shape))
        results[key] = np.array(results[key])[selected_boxes]

    if 'gt_masks' in results: #TODO
        results['gt_masks'] = [results['gt_masks'][i] for i in selected_boxes]
        results['gt_masks'] = [
            mask[ymin:ymax, xmin:xmax]
            for mask in results['gt_masks']
        ]

    keys = ['img']
    for key in keys:
        results[key] = results[key][ymin:ymax, xmin:xmax, :]

    keys = ['gt_bboxes', 'gt_bboxes_ignore']
    for key in keys:
        results[key][:, 0::2] -= xmin
        results[key][:, 1::2] -= ymin

    keys = ['gt_center_line']
    for key in keys:
        results[key][:, :, 0] -= xmin
        results[key][:, :, 1] -= ymin

    keys = ['gt_ori_poly']
    for key in keys:
        ret_coll =[]
        for poly in results[key]:
            poly[:,0] -= xmin
            poly[:,1] -= ymin
            ret_coll.append(poly)
        results[key] = ret_coll

    keys = ['gt_width_line', 'gt_poly_pts']
    for key in keys:
        results[key][:, :, :, 0] -= xmin
        results[key][:, :, :, 1] -= ymin
    return regular_resize(results, crop_size)


def regular_crop(results, crop_size, max_tries, w_array, h_array, w_axis, h_axis, min_crop_side_ratio):
    image = results['img']
    boxes = results['gt_bboxes']

    boxes_ignore = results['gt_bboxes_ignore']

    h, w, c = image.shape
    mask_w = np.arange(w - crop_size)
    mask_h = np.arange(h - crop_size)
    keep_w = np.where(np.logical_and(
        w_array[mask_w] == 0, w_array[mask_w + crop_size - 1] == 0))[0]
    keep_h = np.where(np.logical_and(
        h_array[mask_h] == 0, h_array[mask_h + crop_size - 1] == 0))[0]

    if keep_w.size > 0 and keep_h.size > 0:
        for i in range(max_tries):
            xmin = np.random.choice(keep_w, size=1)[0]
            xmax = xmin + crop_size
            ymin = np.random.choice(keep_h, size=1)[0]
            ymax = ymin + crop_size

            if boxes.shape[0] != 0:
                box_axis_in_area = (boxes[:, 0::2] >= xmin) & (boxes[:, 0::2] <= xmax) \
                    & (boxes[:, 1::2] >= ymin) & (boxes[:, 1::2] <= ymax)
                selected_boxes = np.where(
                    np.sum(box_axis_in_area, axis=1) == 2)[0]
                if len(selected_boxes) > 0:
                    if boxes_ignore.shape[0] !=0:
                        boxes_ignore_in_area = (boxes_ignore[:, 0::2] >= xmin) & (boxes_ignore[:, 0::2] <= xmax) \
                            & (boxes_ignore[:, 1::2] >= ymin) & (boxes_ignore[:, 1::2] <= ymax)
                        selected_boxes_ignore = np.where(
                            np.sum(boxes_ignore_in_area, axis=1) == 2)[0]
                    else:
                        selected_boxes_ignore = []
                    break
            else:
                selected_boxes = []
                if boxes_ignore.shape[0] != 0:
                    boxes_ignore_in_area = (boxes_ignore[:, 0::2] >= xmin) & (boxes_ignore[:, 0::2] <= xmax) \
                                           & (boxes_ignore[:, 1::2] >= ymin) & (boxes_ignore[:, 1::2] <= ymax)
                    selected_boxes_ignore = np.where(
                        np.sum(boxes_ignore_in_area, axis=1) == 2)[0]
                else:
                    selected_boxes_ignore = []
                break
        if i == max_tries-1:
            return random_crop(results, crop_size, max_tries, w_axis, h_axis, min_crop_side_ratio)

        results['gt_bboxes_ignore'] = results['gt_bboxes_ignore'][selected_boxes_ignore]

        keys = ['gt_bboxes', 'gt_labels', 'gt_poly_pts', 'gt_ori_poly', 'gt_center_line', 'gt_width_line', 'gt_line_angle', 'gt_line_length', 'gt_line_x', 'gt_line_y', 'gt_direction', 'gt_transcriptions']
        for key in keys:
            results[key] = np.array(results[key])[selected_boxes]

        if 'gt_masks' in results:
            results['gt_masks'] = [results['gt_masks'][i] for i in selected_boxes]
            results['gt_masks'] =[
                    mask[ymin:ymax, xmin:xmax]
                    for mask in results['gt_masks']
                ]

        keys = ['img']
        for key in keys:
            results[key] = results[key][ymin:ymax, xmin:xmax, :]

        keys = ['gt_bboxes', 'gt_bboxes_ignore']
        for key in keys:
            results[key][:, 0::2] -= xmin
            results[key][:, 1::2] -= ymin

        keys = ['gt_center_line']
        for key in keys:
            results[key][:, :, 0] -= xmin
            results[key][:, :, 1] -= ymin

        keys = ['gt_ori_poly']
        for key in keys:
            ret_coll = []
            for poly in results[key]:
                poly[:, 0] -= xmin
                poly[:, 1] -= ymin
                ret_coll.append(poly)
            results[key] = ret_coll

        keys = ['gt_width_line', 'gt_poly_pts']
        for key in keys:
            results[key][:, :, :, 0] -= xmin
            results[key][:, :, :, 1] -= ymin

        results['img_shape'] = results['img'].shape
        return results
    else:
        return random_crop(results, crop_size, max_tries, w_axis, h_axis, min_crop_side_ratio)

'''
keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore','gt_poly_pts',
                                       'gt_center_line', 'gt_line_angle', 'gt_line_length', 'gt_line_x', 'gt_line_y',
                                       'gt_direction']
'''

class RandomCrop(object):
    def __init__(self, crop_size=640, max_tries=50, min_crop_side_ratio=0.1):
        self.crop_size = crop_size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, results):
        image = results['img']
        h, w, _ = image.shape
        h_array = np.zeros((h), dtype=np.int32)
        w_array = np.zeros((w), dtype=np.int32)

        boxes = results['gt_bboxes']
        boxes_ignore = results['gt_bboxes_ignore']

        for box in np.concatenate((boxes, boxes_ignore), axis=0):
            box = np.round(box, decimals=0).astype(np.int32)
            minx = np.min(box[0::2])
            maxx = np.max(box[0::2])
            w_array[minx:maxx] = 1
            miny = np.min(box[1::2])
            maxy = np.max(box[1::2])
            h_array[miny:maxy] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            # resize image
            return regular_resize(results, self.crop_size)

        if h <= self.crop_size + 1 or w <= self.crop_size + 1:
            return random_crop(results, self.crop_size, self.max_tries, w_axis, h_axis, self.min_crop_side_ratio)
        else:
            return regular_crop(results, self.crop_size, self.max_tries, w_array, h_array, w_axis, h_axis, self.min_crop_side_ratio)

@PIPELINES.register_module
class TextRandomRotate(object):
    def __init__(self, rotate_ratio, BBox_type, max_rotate_angle=45, num_width_line=9):
        self.rotate_ratio = rotate_ratio
        self.max_rotate_angle = max_rotate_angle
        self.num_width_line = num_width_line
        self.BBox_type = BBox_type

    def __call__(self, results):
        if random.random() < self.rotate_ratio:
            rotate_angle = random.uniform(-1 * self.max_rotate_angle, self.max_rotate_angle)
            results['rotate_angle'] = rotate_angle
            # print('rotate_angle {}'.format(rotate_angle))
            height, width, _ = results['img_shape']
            # print('img shape pre rotate {}'.format(results['img_shape']))
            img = results['img']
            ## get the minimal rect to cover the rotated image
            img_box = [[[0, 0], [width, 0], [width, height], [0, height]]]
            rotated_img_box = self._quad2minrect(self._rotate_polygons(img_box, rotate_angle, (width / 2, height / 2)))
            r_height = int(max(rotated_img_box[0][3], rotated_img_box[0][1]) - min(rotated_img_box[0][3], rotated_img_box[0][1]))
            r_width = int(max(rotated_img_box[0][2], rotated_img_box[0][0]) - min(rotated_img_box[0][2], rotated_img_box[0][0]))
            r_height = max(r_height, height + 1)
            r_width = max(r_width, width + 1)

            ## padding img and mask
            img_padding = np.zeros((r_height, r_width, 3))
            start_h, start_w = int((r_height - height) / 2.0), int((r_width - width) / 2.0)
            end_h, end_w = start_h + height, start_w + width
            img_padding[start_h:end_h, start_w:end_w, :] = img

            M = cv2.getRotationMatrix2D((r_width / 2, r_height / 2), rotate_angle, 1)

            results['img'] = cv2.warpAffine(img_padding, M, (r_width, r_height))
            results['img_shape'] = results['img'].shape
            # print('img shape after rotate {}'.format(results['img_shape']))

            if 'gt_masks' in results:
                rotated_gt_masks = []
                for mask in results['gt_masks']:
                    mask_padding = np.zeros((r_height, r_width, 1))
                    # print('mask pre rotate {}'.format(mask.shape))
                    # print('mask pre rotate {}'.format(mask))
                    mask_padding[start_h:end_h, start_w:end_w, :] = np.expand_dims(mask, axis=-1)
                    mask = cv2.warpAffine(mask_padding, M, (r_width, r_height))
                    # print('mask after rotate {}'.format(mask.shape))
                    rotated_gt_masks.append(mask)
                results['gt_masks'] = rotated_gt_masks

            gt_bboxes = []
            # gt_labels = []
            # gt_bboxes_ignore = []
            gt_ori_poly = []
            # gt_masks_ann = []
            gt_center_line = []
            gt_width_line = []
            gt_line_angle = []
            gt_line_length = []
            gt_line_x = []
            gt_line_y = []
            gt_direction = []
            gt_poly_pts = []
            # gt_transcriptions = []
            # for key in ['center_line', 'ori_poly', 'width_line', 'line_angle', 'line_length', 'line_x', 'line_y', 'direction', 'poly_pts', 'transcriptions']:
            none_to_ignore = []
            none_to_del_ind = []
            for ind, poly in enumerate(results['gt_ori_poly']):
                poly[:, 0] += start_w
                poly[:, 1] += start_h
                poly = np.concatenate((poly, np.ones((poly.shape[0], 1))), axis=-1).transpose(1, 0)
                poly = np.matmul(M, poly).transpose(1, 0)
                poly_label = self.BBox_type(poly.astype(np.int), num_width_line=self.num_width_line).gen_poly_label()
                if poly_label is None:
                    none_to_ignore.append(poly)
                    none_to_del_ind.append(ind)
                    continue
                gt_ori_poly.append(poly)
                gt_bboxes.append(poly_label['bbox'])
                # gt_labels.append(self.cat2label[script])
                # gt_ori_poly.append(poly_label['poly'])
                # gt_masks_ann.append([poly_label['poly'].flatten().tolist()])
                gt_center_line.append(poly_label['center_line'])
                gt_width_line.append(poly_label['width_line'])
                gt_line_angle.append(poly_label['line_angle'])
                gt_line_length.append(poly_label['line_length'])
                gt_line_x.append(poly_label['line_x'])
                gt_line_y.append(poly_label['line_y'])
                gt_direction.append(poly_label['direction'])
                gt_poly_pts.append(poly_label['origin_poly_pts'])
                # gt_transcriptions.append(transcription)

            results['gt_center_line'] = gt_center_line
            results['gt_ori_poly'] = gt_ori_poly
            results['gt_width_line'] = gt_width_line
            results['gt_line_angle'] = gt_line_angle
            results['gt_line_length'] = gt_line_length
            results['gt_line_x'] = gt_line_x
            results['gt_line_y'] = gt_line_y
            results['gt_direction'] = gt_direction
            results['gt_poly_pts'] = gt_poly_pts

            # if gt_bboxes:
            #     results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            #     # gt_labels = np.array(gt_labels, dtype=np.int64)
            #     # gt_masks_ann = gt_masks_ann
            # else:
            #     results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            #     # gt_labels = np.array([], dtype=np.int64)
            #     # gt_masks_ann = [[[0, 0, 0, 0]]] #TODO

            if none_to_del_ind:
                results['gt_masks'] = [mask for i, mask in enumerate(results['gt_masks']) if i not in none_to_del_ind]
                results['gt_transcriptions'] = [transcription for i, transcription in enumerate(results['gt_transcriptions']) if i not in none_to_del_ind]
                results['gt_labels'] = [label for i, label in enumerate(results['gt_labels']) if i not in none_to_del_ind]

            if gt_bboxes:
                results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32)
                results['gt_labels'] = np.array(results['gt_labels'], dtype=np.int64)
                # gt_masks_ann = gt_masks_ann
            else:
                results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                results['gt_labels'] = np.array([], dtype=np.int64)
                # gt_masks_ann = [[[0, 0, 0, 0]]] #TODO

            rotated_bboxes_ignore = []
            for box in results['gt_bboxes_ignore']:
                reformat_box = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
                reformat_box[:, 0] += start_w
                reformat_box[:, 1] += start_h
                reformat_box = np.concatenate((reformat_box, np.ones((reformat_box.shape[0], 1))), axis=-1).transpose(1, 0)
                reformat_box = np.matmul(M, reformat_box).transpose(1, 0)
                rotated_bboxes_ignore.append((np.min(reformat_box[:, 0]), np.min(reformat_box[:, 1]),
                                              np.max(reformat_box[:, 0]), np.max(reformat_box[:, 1])))
            for poly in none_to_ignore:
                rotated_bboxes_ignore.append((np.min(poly[:, 0]), np.min(poly[:, 1]),
                                              np.max(poly[:, 0]), np.max(poly[:, 1])))
            if rotated_bboxes_ignore:
                results['gt_bboxes_ignore'] = np.array(rotated_bboxes_ignore, dtype=np.float32)
            else:
                results['gt_bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
            return results
        else:
            results['rotate_angle'] = 0
            return results

    def _quad2minrect(seif, boxes):
        ## trans a quad(N*4) to a rectangle(N*4) which has miniual area to cover it
        return np.hstack((boxes[:, ::2].min(axis=1).reshape((-1, 1)), boxes[:, 1::2].min(axis=1).reshape((-1, 1)), boxes[:, ::2].max(axis=1).reshape((-1, 1)),
                          boxes[:, 1::2].max(axis=1).reshape((-1, 1))))

    def _boxlist2quads(self, boxlist):
        res = np.zeros((len(boxlist), 8))
        for i, box in enumerate(boxlist):
            # print(box)
            res[i] = np.array([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]])
        return res

    def _rotate_polygons(self, polygons, angle, r_c):
        rotate_boxes_list = []
        for poly in polygons:
            box = Polygon(poly)
            rbox = affinity.rotate(box, angle, r_c)
            if len(list(rbox.exterior.coords)) < 5:
                print('img_box_ori:', poly)
                print('img_box_rotated:', rbox)
            # assert(len(list(rbox.exterior.coords))>=5)
            rotate_boxes_list.append(rbox.boundary.coords[:-1])
        res = self._boxlist2quads(rotate_boxes_list)
        return res
