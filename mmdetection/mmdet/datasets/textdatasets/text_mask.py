import os
from os import path as osp
import cv2
import gc
from abc import ABCMeta, abstractmethod
import imageio
import json
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from ..custom import CustomDataset
from ..registry import DATASETS

@DATASETS.register_module
class CurveTextDataset_MASK(CustomDataset):
    CLASSES = ('background', 'text')

    IGNORE_IM_FNS = dict(
        # (im_fn, action, counterclockwise angle)
        train=[],
        val=[],
        test=[]
    )

    @abstractmethod
    def load_bboxes(self, gt_fp):
        pass

    @abstractmethod
    def format_gt_fn(self, im_fn):
        pass

    def prepare_annotations(self, imgs_dir, gts_dir, ann_file, regen_ann=False):
        if 'train' in ann_file: dataset_type = 'train'
        elif 'val' in ann_file: dataset_type = 'val'
        elif 'test' in ann_file: 
            dataset_type = 'test'
            os.makedirs(gts_dir, exist_ok=True)

        if not osp.exists(ann_file) or regen_ann:
            im_fns = sorted([im_fn for im_fn in os.listdir(imgs_dir) if self.is_image_file(im_fn)])
            gt_fns = sorted([gt_fn for gt_fn in os.listdir(gts_dir) if gt_fn.endswith('.txt')])
            if dataset_type == 'test': gt_fns = [None] * len(im_fns)
            assert dataset_type == 'test' or \
                   (len(im_fns) != 0 and len(gt_fns) != 0 and
                    all([self.format_gt_fn(im_fn) == gt_fn for im_fn, gt_fn in zip(im_fns, gt_fns)]))

            print('Find {} images of train dataset -- {} and loading...'.format(len(im_fns), imgs_dir))
            anns_list, del_idx_list = [], []
            for idx, (im_fn, gt_fn) in tqdm(enumerate(zip(im_fns, gt_fns))):
                im = self.read_image(osp.join(imgs_dir, im_fn))
                if im is None:
                    del_idx_list.append(idx)
                    continue

                height, width, channel = im.shape
                if dataset_type == 'test':
                    bboxes = []
                else:
                    bboxes = self.load_bboxes(osp.join(gts_dir, gt_fn))
                ann = dict(
                    filename=im_fn,
                    width=width,
                    height=height,
                    bboxes=bboxes
                )
                anns_list.append(ann)

            for idx in del_idx_list:
                im_fns.remove(im_fns[idx])
                gt_fns.remove(gt_fns[idx])

            for im_fn, action, angle in self.IGNORE_IM_FNS[dataset_type]:
                if im_fn not in im_fns:
                    continue

                idx = im_fns.index(im_fn)
                assert im_fn == anns_list[idx]['filename'], im_fn + ' ' + anns_list[idx]['filename']
                if action == 'rotate':
                    ann = anns_list[idx]
                    height, width = ann['height'], ann['width']
                    new_bboxes = []
                    for bbox_str in ann['bboxes']:
                        splits = bbox_str.split(',')
                        pts = np.array([int(x) for x in splits[:8]]).reshape(-1, 2)
                        if angle == 90:
                            new_pts = np.array([(width-y, x) for x,y in pts]).reshape(-1).tolist()
                        elif angle == 180:
                            new_pts = np.array([(width-x, height-y) for x,y in pts]).reshape(-1).tolist()
                        else:
                            new_pts = pts.reshape(-1).tolist()
                        new_bbox = ','.join([','.join([str(x) for x in new_pts]), ','.join(splits[8:])])
                        new_bboxes.append(new_bbox)
                    ann['bboxes'] = new_bboxes
                elif action == 'delete':
                    del anns_list[idx]
                    del im_fns[idx]
                    del gt_fns[idx]
            
            with open(ann_file, 'w', encoding='utf-8') as f:
                json.dump(anns_list, f)
        else:
            with open(ann_file, 'r', encoding='utf-8') as f:
                anns_list = json.load(f)
        print('Load {} images of train dataset -- {}'.format(len(anns_list), imgs_dir))

        return anns_list

    def load_annotations(self, ann_file):
        gts_dir = osp.dirname(ann_file)
        imgs_dir = osp.join(osp.dirname(gts_dir), 'images')
        
        self.ann_list = self.prepare_annotations(imgs_dir, gts_dir, ann_file, regen_ann=self.regen_ann)
        self.cat2label = {
            cat_id : i
            for i, cat_id in enumerate(self.CLASSES)
        }

        return self.ann_list

    def get_ann_info(self, idx):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        height, width = self.ann_list[idx]['height'], self.ann_list[idx]['width']
        for bbox_str in self.ann_list[idx]['bboxes']:
            splits = bbox_str.split(',')
            poly, script, transcription = splits[:-2], splits[-2], splits[-1]
            poly = np.array([int(x) for x in poly]).reshape(-1,2)
            bbox = (np.min(poly[:, 0]), np.min(poly[:, 1]),
                    np.max(poly[:, 0]), np.max(poly[:, 1]))
            # poly_mask = np.zeros((height, width), dtype=np.uint8)
            # cv2.fillPoly(poly_mask, poly[np.newaxis, :, :], 1)

            if script =='#':
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[script])
                gt_masks_ann.append([poly.reshape(-1).tolist()])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_masks_ann = gt_masks_ann
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_masks_ann = [[[0, 0, 0, 0]]]

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann)

        return ann


    def is_image_file(self, im_fn):
        exts = ['.jpg', '.JPG', '.jpeg', '.png', '.PNG', '.gif']
        return any([im_fn.endswith(ext) for ext in exts])

    def read_image(self, im_fp):
        if im_fp.endswith('.gif'):
            im = imageio.mimread(im_fp)
            if im is not None:
                im = np.array(im)[0][:, :, 0:3]
        else:
            im = cv2.imread(im_fp)
        
        return im

@DATASETS.register_module
class MLT2017Dataset_MASK(CurveTextDataset_MASK):

    def __init__(self, regen_ann=False, num_width_line = 5, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line

        super(MLT2017Dataset_MASK, self).__init__(**kwargs)

    CLASSES = ('background', 'text')
    # 'Mixed', 'None'
    # CLASSES = ('background', 'Symbols', 'Arabic', 'Latin', 'Chinese', 'Japanese', 'Korean', 'Bangla')

    IGNORE_IM_FNS = dict(
        # (im_fn, action, counterclockwise angle)
        train=[
            ('img_7.jpg', 'rotate', 90),
            ('img_3591.jpg', 'delete', -1),
            ('img_5554.jpg', 'delete', -1),
            ('img_5741.jpg', 'delete', -1),
        ],
        val=[
            ('img_144.jpg', 'rotate', 180),
            ('img_150.jpg', 'rotate', 90),
            ('img_153.jpg', 'rotate', 90),
            ('img_154.jpg', 'rotate', 90),
            ('img_155.jpg', 'rotate', 90),
            ('img_156.jpg', 'rotate', 90),
            ('img_157.jpg', 'rotate', 90),
            ('img_158.jpg', 'rotate', 90),
            ('img_159.jpg', 'rotate', 90),
            ('img_160.jpg', 'rotate', 90),
            ('img_161.jpg', 'rotate', 90),
            ('img_169.jpg', 'rotate', 90),
            ('img_173.jpg', 'rotate', 90),
            ('img_174.jpg', 'rotate', 90),
            ('img_176.jpg', 'rotate', 90),
            ('img_838.jpg', 'delete', -1),
            ('img_937.jpg', 'delete', -1),
        ],
        test=[]
    )

    def format_gt_fn(self, im_fn):
        return 'gt_' + osp.splitext(im_fn)[0] + self.gt_fn_ext

    def load_bboxes(self, gt_fp):
        with open(gt_fp, encoding='utf-8-sig') as f:
            bboxes = []
            for line in f.readlines():
                splits = line.strip('\n').split(',')
                poly, script, transcription = splits[:8], splits[8], splits[9]
                script = '#' if transcription == '###' or script in ['Mixed', 'None'] else 'text'
                bboxes.append(','.join(poly + [script, transcription]))

        return bboxes


@DATASETS.register_module
class ICDAR2015Dataset_MASK(CurveTextDataset_MASK):

    def __init__(self, regen_ann=False, num_width_line = 5, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line

        super(ICDAR2015Dataset_MASK, self).__init__(**kwargs)

    CLASSES = ('background', 'text')

    IGNORE_IM_FNS = dict(
        # (im_fn, action, counterclockwise angle)
        train=[],
        val=[],
        test=[]
    )

    def format_gt_fn(self, im_fn):
        return im_fn + self.gt_fn_ext

    def load_bboxes(self, gt_fp):
        with open(gt_fp, encoding='utf-8-sig') as f:
            bboxes = []
            for line in f.readlines():
                splits = line.strip('\n').split(',')
                poly, transcription = splits[:8], splits[8]
                script = '#' if transcription == '###' else 'text'
                bboxes.append(','.join(poly + [script, transcription]))

        return bboxes


@DATASETS.register_module
class TotalTextDataset_MASK(CurveTextDataset_MASK):

    def __init__(self, regen_ann=False, num_width_line = 5, **kwargs):
        self.gt_fn_ext = '.mat'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line

        super(TotalTextDataset_MASK, self).__init__(**kwargs)

    CLASSES = ('background', 'text')

    IGNORE_IM_FNS = dict(
        # (im_fn, action, counterclockwise angle)
        train=[],
        val=[],
        test=[]
    )

    def format_gt_fn(self, im_fn):
        return 'gt_' + osp.splitext(im_fn)[0] + self.gt_fn_ext

    def load_bboxes(self, gt_fp):
        bboxes = []
        gt_mat = sio.loadmat(gt_fp)
        for item in gt_mat['gt']:
            xs, ys, transcription, script = item[1], item[3], item[4], item[5]
            bbox = np.stack((xs, ys), axis=-1).flatten().tolist()
            script = '#' if script == '#' else 'text'
            line = ','.join([str(x) for x in bbox] + [script, transcription[0]])
            bboxes.append(line)

        return bboxes


@DATASETS.register_module
class CTW1500Dataset_MASK(CurveTextDataset_MASK):

    def __init__(self, regen_ann=False, num_width_line = 5, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line

        super(CTW1500Dataset_MASK, self).__init__(**kwargs)

    CLASSES = ('background', 'text')

    IGNORE_IM_FNS = dict(
        # (im_fn, action, counterclockwise angle)
        train=[],
        val=[],
        test=[]
    )

    def format_gt_fn(self, im_fn):
        return osp.splitext(im_fn)[0] + self.gt_fn_ext

    def load_bboxes(self, gt_fp):
        bboxes = []
        with open(gt_fp, encoding='utf-8') as f:
            for line in f.readlines():
                splits = line.strip('\n').split(',')
                poly, transcription = [int(x) for x in splits], 'text'
                x, y, w, h, offsets = poly[0], poly[1], poly[2], poly[3], poly[4:]
                bbox = np.array([x, y] * (len(offsets) // 2)) + np.array(offsets)
                script = '#' if transcription == '###' else 'text'
                bboxes.append(','.join([str(x) for x in bbox] + [script, transcription]))

        return bboxes