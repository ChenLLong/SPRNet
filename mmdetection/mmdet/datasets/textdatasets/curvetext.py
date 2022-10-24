import os
from os import path as osp
from abc import ABCMeta, abstractmethod
import cv2
import imageio
import json
import gc
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import xml
from xml.dom import minidom
import torch

from ..custom import CustomDataset
from ..registry import DATASETS
from .polybbox import PolyBBox

from torch.multiprocessing import Manager, Pool
import torch.multiprocessing as mp


@DATASETS.register_module
class CurveTextDataset(CustomDataset):
    CLASSES = ('background', 'text')
    CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
                '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
                '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

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
        if 'train' in ann_file:
            dataset_type = 'train'
        elif 'val' in ann_file:
            dataset_type = 'val'
        elif 'test' in ann_file:
            dataset_type = 'test'
            os.makedirs(gts_dir, exist_ok=True)
        else:
            raise ValueError('ann_file must contain (train, val, test)!')

        if not osp.exists(ann_file) or regen_ann:
            im_fns = sorted([im_fn for im_fn in os.listdir(imgs_dir) if self.is_image_file(im_fn)])
            gt_fns = sorted([gt_fn for gt_fn in os.listdir(gts_dir) if gt_fn.endswith(self.gt_fn_ext)])
            if dataset_type == 'test': gt_fns = [None] * len(im_fns)
            assert dataset_type == 'test' or \
                   (len(im_fns) != 0 and len(gt_fns) != 0 and
                    all([self.format_gt_fn(im_fn) == gt_fn for im_fn, gt_fn in zip(im_fns, gt_fns)]))

            print('Find {} images of train dataset -- {} and loading...'.format(len(im_fns), imgs_dir))
            anns_list, del_idx_list = [], []
            gc.disable()
            for idx, (im_fn, gt_fn) in tqdm(enumerate(zip(im_fns, gt_fns))):
                im = self.read_image(osp.join(imgs_dir, im_fn))
                if im is None:
                    del_idx_list.append(idx)
                    continue

                height, width, channel = im.shape
                bboxes = []
                if dataset_type != 'test':
                    bboxes = self.load_bboxes(osp.join(gts_dir, gt_fn))
                ann = dict(
                    filename=im_fn,
                    width=width,
                    height=height,
                    bboxes=bboxes
                )
                anns_list.append(ann)
            gc.enable()
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
                    for bbox_item in ann['bboxes']:
                        splits = bbox_item['bbox']
                        pts = np.array([int(x) for x in splits]).reshape(-1, 2)
                        if angle == 90:
                            new_pts = np.array([(width - y, x) for x, y in pts]).flatten().tolist()
                        elif angle == 180:
                            new_pts = np.array([(width - x, height - y) for x, y in pts]).flatten().tolist()
                        elif angle == 270:
                            new_pts = np.array([(y, height - x) for x, y in pts]).flatten().tolist()
                        else:
                            new_pts = pts.flatten().tolist()
                        new_bbox_item = dict(bbox=[str(x) for x in new_pts], script=bbox_item['script'], transcription=bbox_item['transcription'])
                        new_bboxes.append(new_bbox_item)
                    ann['bboxes'] = new_bboxes
                    anns_list[idx] = ann
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
        gc.disable()
        for idx, ann in tqdm(enumerate(anns_list)):
            ann['poly_label'] = []
            for bbox_item in ann['bboxes']:
                #bbox=poly, script=script, transcription=transcription
                poly, script, transcription = bbox_item['bbox'], bbox_item['script'], bbox_item['transcription']
                poly = np.array([int(x) for x in poly]).reshape(-1, 2)
                ann['poly_label'].append(self.BBox_type(poly, num_width_line= self.num_width_line).gen_poly_label())
        gc.enable()
        return anns_list

    def load_annotations(self, ann_file):
        gts_dir = osp.dirname(ann_file)
        imgs_dir = self.img_prefix

        self.ann_list = self.prepare_annotations(imgs_dir, gts_dir, ann_file, regen_ann=self.regen_ann)
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.CLASSES)
        }

        return self.ann_list

    def get_ann_info(self, idx):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_ori_poly = []
        gt_masks_ann = []
        gt_center_line = []
        gt_width_line=[]
        gt_line_angle = []
        gt_line_length = []
        gt_line_x = []
        gt_line_y = []
        gt_direction = []
        gt_poly_pts = []
        gt_transcriptions = []

        for i, bbox_item in enumerate(self.ann_list[idx]['bboxes']):
            poly, script, transcription = bbox_item['bbox'], bbox_item['script'], bbox_item['transcription']
            poly = np.array([int(x) for x in poly]).reshape(-1, 2)
            poly_label = self.ann_list[idx]['poly_label'][i]
            if poly_label is None:
                # gt_bboxes_ignore.append((np.min(poly[:, 0]), np.min(poly[:, 1]),
                #                         np.max(poly[:, 0]), np.max(poly[:, 1])))
                gt_bboxes_ignore.append(poly)
            elif script == '#':
                # gt_bboxes_ignore.append(poly_label['bbox'])
                gt_bboxes_ignore.append(poly)
            else:
                gt_bboxes.append(poly_label['bbox'])
                gt_labels.append(self.cat2label[script])
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
                gt_transcriptions.append(transcription)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_masks_ann = gt_masks_ann
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_masks_ann = [[[0, 0, 0, 0, 0, 0]]]

        # if gt_bboxes_ignore:
        #     gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        # else:
        #     gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

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

        return ann

    def pre_pipeline(self, results):
        super(CurveTextDataset, self).pre_pipeline(results)
        results['poly_fields'] = []

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
class MLT2017Dataset(CurveTextDataset):

    def __init__(self, regen_ann=False, num_width_line = 5, BBox_type=PolyBBox, ratio=1, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line
        self.BBox_type=BBox_type

        super(MLT2017Dataset, self).__init__(**kwargs)

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
                transcription = 'text'
                bboxes.append(dict(bbox=poly, script=script, transcription=transcription))

        return bboxes


@DATASETS.register_module
class ICDAR2015Dataset(CurveTextDataset):

    def __init__(self, regen_ann=False, num_width_line = 5, BBox_type=PolyBBox, ratio=1, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line
        self.BBox_type = BBox_type
        super(ICDAR2015Dataset, self).__init__(**kwargs)

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
                bboxes.append(dict(bbox=poly, script=script, transcription=transcription))

        return bboxes


@DATASETS.register_module
class ICDAR2013Dataset(CurveTextDataset):

    def __init__(self, regen_ann=False, num_width_line = 5, BBox_type=PolyBBox, ratio=1, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line
        self.BBox_type = BBox_type
        super(ICDAR2013Dataset, self).__init__(**kwargs)

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
        with open(gt_fp, encoding='utf-8-sig') as f:
            bboxes = []
            for line in f.readlines():
                splits = line.strip('\n').split(' ')
                poly, transcription = splits[:4], splits[4].replace('"', '')
                poly = [poly[0], poly[1], poly[2], poly[1], poly[2], poly[3], poly[0], poly[3]]
                script = '#' if transcription == '###' else 'text'
                bboxes.append(dict(bbox=poly, script=script, transcription=transcription))

        return bboxes

@DATASETS.register_module
class COCOTextDataset(CurveTextDataset):

    def __init__(self, regen_ann=False, num_width_line = 5, BBox_type=PolyBBox, ratio=1, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line
        self.BBox_type = BBox_type
        super(COCOTextDataset, self).__init__(**kwargs)

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
        with open(gt_fp, encoding='utf-8-sig') as f:
            bboxes = []
            for line in f.readlines():
                splits = line.strip('\n').split(',')
                poly, transcription = splits[:8], splits[8]
                script = '#' if transcription == '###' else 'text'
                bboxes.append(dict(bbox=poly, script=script, transcription=transcription))

        return bboxes

@DATASETS.register_module
class SCUTDataset(CurveTextDataset):

    def __init__(self, regen_ann=False, num_width_line = 5, BBox_type=PolyBBox, ratio=1, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line
        self.BBox_type = BBox_type
        super(SCUTDataset, self).__init__(**kwargs)

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
                bboxes.append(dict(bbox=[float(x) for x in poly], script=script, transcription=transcription))

        return bboxes

@DATASETS.register_module
class TotalTextDataset(CurveTextDataset):

    def __init__(self, regen_ann=False, num_width_line=5, BBox_type=PolyBBox, ratio=1, **kwargs):
        self.gt_fn_ext = '.mat'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line
        self.BBox_type = BBox_type
        super(TotalTextDataset, self).__init__(**kwargs)

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
            # c=curve; h=horizontal; m=multi-oriented; #=dont care
            xs, ys, transcription, script = item[1], item[3], item[4], item[5]
            bbox = np.stack((xs, ys), axis=-1).flatten().tolist()
            script = '#' if script == '#' else 'text'
            # line = ','.join([str(x) for x in bbox] + [script, transcription[0]])
            bboxes.append(dict(bbox=[str(x) for x in bbox], script=script, transcription=transcription[0]))

        return bboxes


@DATASETS.register_module
class CTW1500Dataset(CurveTextDataset):

    def __init__(self, regen_ann=False, num_width_line=5, BBox_type=PolyBBox, ratio=1, **kwargs):
        self.gt_fn_ext = '.xml'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line
        self.BBox_type = BBox_type
        super(CTW1500Dataset, self).__init__(**kwargs)

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
        if gt_fp.endswith('txt'):
            with open(gt_fp, encoding='utf-8') as f:
                if 'e2e' not in gt_fp:
                    for line in f.readlines():
                        splits = line.strip('\n').split(',')
                        poly, transcription = [int(x) for x in splits], 'text'
                        x, y, w, h, offsets = poly[0], poly[1], poly[2], poly[3], poly[4:]
                        bbox = np.array([x, y] * (len(offsets) // 2)) + np.array(offsets)
                        script = '#' if transcription == '###' else 'text'
                        bboxes.append(dict(bbox=[str(x) for x in bbox], script=script, transcription=transcription))
                else:
                    for idx, line in enumerate(f.readlines()):
                        splits = line.strip('\n').split(',')
                        transcription = splits[-1]
                        bbox = [int(x) for x in splits[:-1]]
                        script = '#' if transcription == '###' else 'text'
                        bboxes.append(dict(bbox=[str(x) for x in bbox], script=script, transcription=transcription))
        elif gt_fp.endswith('xml'):
            dom = xml.dom.minidom.parse(gt_fp)
            bboxes_xml = dom.getElementsByTagName('box')
            for idx, bbox_item in enumerate(bboxes_xml):
                transcription = bbox_item.getElementsByTagName('label')[0].firstChild.data
                bbox = bbox_item.getElementsByTagName('segs')[0].firstChild.data.split(',')
                script = '#' if transcription == '###' else 'text'
                bboxes.append(dict(bbox=[str(x) for x in bbox], script=script, transcription=transcription))
        return bboxes


@DATASETS.register_module
class ICDAR19ArtDataset(CurveTextDataset):

    def __init__(self, regen_ann=False, num_width_line = 5, ratio=1, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.gt_name = 'train_labels.json'
        self.num_width_line = num_width_line

        super(ICDAR19ArtDataset, self).__init__(**kwargs)

    CLASSES = ('background', 'text')

    IGNORE_IM_FNS = dict(
        # (im_fn, action, counterclockwise angle)
        train=[
            ('gt_41.jpg', 'rotate', 90),
            ('gt_941.jpg', 'rotate', 270),
            ('gt_1458.jpg', 'rotate', 90),
            ('gt_1491.jpg', 'rotate', 90),
            ('gt_2440.jpg', 'rotate', 90),
            ('gt_4639.jpg', 'rotate', 90),
            ('gt_5194.jpg', 'rotate', 90),
        ],
        val=[],
        test=[]
    )

    # def format_gt_fn(self, im_fn):
    #     return osp.splitext(im_fn)[0] + self.gt_fn_ext

    def load_bboxes(self, im_fn):
        img_id = im_fn.split('.')[0]
        instances = self.gts[img_id]
        bboxes = []
        for instance in instances:
            poly = instance['points']
            if instance['illegibility']:
                poly, transcription = np.array(poly).flatten().tolist(), '###'
            else:
                poly, transcription = np.array(poly).flatten().tolist(), 'text'
            script = '#' if transcription == '###' else 'text'
            bboxes.append(','.join([ str(i) for i in poly ] + [script, transcription]))
        return bboxes

    def prepare_annotations(self, imgs_dir, gts_dir, ann_file, regen_ann=False):
        if 'train' in ann_file:
            dataset_type = 'train'
        elif 'val' in ann_file:
            dataset_type = 'val'
        elif 'test' in ann_file:
            dataset_type = 'test'
            os.makedirs(gts_dir, exist_ok=True)
        else:
            raise ValueError('ann_file must contain (train, val, test)!')

        image_dete_id = dict()
        imgs_detelt_dir = '/public/scene_text/total_text/test/images'
        with open('/home/sufeng2/sjh/Code/mmdetection/mmdet/datasets/textdatasets/Total_Text_ID_vs_ArT_ID.list', 'r') as f:
            for line in f.readlines():
                ids = line.split(' ')
                id_totaltext_dete = ids[0].strip('.txt').strip('poly_gt_')
                id_ic19art_dete = ids[1].strip('.txt\n')
                image_dete_id[id_totaltext_dete] = id_ic19art_dete
        im_fns = sorted([im_fn for im_fn in os.listdir(imgs_detelt_dir) if self.is_image_file(im_fn)])

        for im_fn in im_fns:
            id = image_dete_id[im_fn.split('.')[0]]
            self.IGNORE_IM_FNS['train'].append((id + '.jpg', 'delete', -1))

        if not osp.exists(ann_file) or regen_ann:
            if osp.exists(osp.join(gts_dir, self.gt_name)):
                with open(osp.join(gts_dir, self.gt_name), 'r') as load_f:
                    self.gts = json.load(load_f)
            im_fns = sorted([im_fn for im_fn in os.listdir(imgs_dir) if self.is_image_file(im_fn)])

            print('Find {} images of train dataset -- {} and loading...'.format(len(im_fns), imgs_dir))
            anns_list, del_idx_list = [], []
            gc.disable()
            for idx, im_fn in tqdm(enumerate(im_fns)):
                im = self.read_image(osp.join(imgs_dir, im_fn))
                if im is None:
                    del_idx_list.append(idx)
                    continue

                height, width, channel = im.shape
                bboxes = []
                if dataset_type != 'test':
                    bboxes = self.load_bboxes(im_fn)
                ann = dict(
                    filename=im_fn,
                    width=width,
                    height=height,
                    bboxes=bboxes
                )
                anns_list.append(ann)
            gc.enable()
            for idx in del_idx_list:
                im_fns.remove(im_fns[idx])
                # gt_fns.remove(gt_fns[idx])

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
                            new_pts = np.array([(width - y, x) for x, y in pts]).flatten().tolist()
                        elif angle == 180:
                            new_pts = np.array([(width - x, height - y) for x, y in pts]).flatten().tolist()
                        elif angle == 270:
                            new_pts = np.array([(y, height - x) for x, y in pts]).flatten().tolist()
                        else:
                            new_pts = pts.flatten().tolist()
                        new_bbox = ','.join([','.join([str(x) for x in new_pts]), ','.join(splits[8:])])
                        new_bboxes.append(new_bbox)
                    ann['bboxes'] = new_bboxes
                elif action == 'delete':
                    del anns_list[idx]
                    del im_fns[idx]

            with open(ann_file, 'w', encoding='utf-8') as f:
                json.dump(anns_list, f)
        else:
            with open(ann_file, 'r', encoding='utf-8') as f:
                anns_list = json.load(f)
        print('Load {} images of train dataset -- {}'.format(len(anns_list), imgs_dir))
        gc.disable()
        for idx, ann in tqdm(enumerate(anns_list)):
            ann['poly_label'] = []
            for bbox_str in ann['bboxes']:
                splits = bbox_str.split(',')
                poly, script, transcription = splits[:-2], splits[-2], splits[-1]
                poly = np.array([int(x) for x in poly]).reshape(-1, 2)
                ann['poly_label'].append(PolyBBox(poly).gen_poly_label())
        gc.enable()
        return anns_list


@DATASETS.register_module
class SynthTextDataset(CurveTextDataset):
    def __init__(self, regen_ann=False, num_width_line=5, BBox_type=PolyBBox, ratio=1, **kwargs):
        self.gt_fn_ext = '.txt'
        self.regen_ann = regen_ann
        self.num_width_line = num_width_line
        self.gt_names = 'gt.mat'
        self.BBox_type = BBox_type
        super(SynthTextDataset, self).__init__(**kwargs)

    CLASSES = ('background', 'text')

    IGNORE_IM_FNS = dict(
        # (im_fn, action, counterclockwise angle)
        train=[],
        val=[],
        test=[]
    )

    def load_bboxes(self, idx, im_fn):
        instances = self.gts['wordBB'][0][idx]
        transcriptions = '\n'.join(self.gts['txt'][0][idx]).split()
        bboxes = []
        instance_shape = instances.shape
        if len(instance_shape) == 2:
            instances = np.expand_dims(instances, axis=-1)
        elif len(instance_shape) != 3:
            print('shape invaild:{}--{}--{}'.format(idx, instance_shape, im_fn))
            return bboxes
        for idx in range(instances.shape[-1]):
            instance = instances[:, :, idx]
            poly = instance.transpose(1, 0)
            poly = np.array(poly)
            poly, transcription = poly.flatten().tolist(), transcriptions[idx]
            script = '#' if transcription == '###' else 'text'
            bboxes.append(dict(bbox=list(map(str, poly)), script=script, transcription=transcription))
        return bboxes

    def single_process(self, queue, imgs_dir):
        while True:
            if not queue.empty():
                idx, im_fn = queue.get()
                im = self.read_image(osp.join(imgs_dir, im_fn))

                if im is None:
                    self.del_idx_list.append(idx)
                    continue

                height, width, channel = im.shape

                bboxes = self.load_bboxes(idx, im_fn)
                ann = dict(
                    filename=im_fn,
                    width=width,
                    height=height,
                    bboxes=bboxes
                )
                self.anns_list.append(ann)

    def prepare_annotations(self, imgs_dir, gts_dir, ann_file, regen_ann=False, num_process=20):
        if 'train' in ann_file:
            dataset_type = 'train'
        elif 'val' in ann_file:
            dataset_type = 'val'
        elif 'test' in ann_file:
            dataset_type = 'test'
            os.makedirs(gts_dir, exist_ok=True)
        else:
            raise ValueError('ann_file must contain (train, val, test)!')

        if not osp.exists(ann_file) or regen_ann:
            print(osp.join(gts_dir, self.gt_names))
            if osp.exists(osp.join(gts_dir, self.gt_names)):
                self.gts = sio.loadmat(osp.join(gts_dir, self.gt_names))
                im_fns = self.gts['imnames'][0]

            print('Find {} images of train dataset -- {} and loading...'.format(len(im_fns), imgs_dir))
            self.anns_list, self.del_idx_list = mp.Manager().list([]), mp.Manager().list([])

            processes = []
            manager = Manager()
            queue = manager.Queue(maxsize=60)
            for _ in range(num_process):
                p = mp.Process(target=self.single_process, args=(queue, imgs_dir))
                p.start()
                processes.append(p)

            for idx, im_fn in tqdm(enumerate(im_fns)):
                queue.put((idx, im_fn[0]))
            #     im = self.read_image(osp.join(imgs_dir, im_fn[0]))
            #     if im is None:
            #         del_idx_list.append(idx)
            #         continue
            #
            #     height, width, channel = im.shape
            #     bboxes = []
            #     if dataset_type != 'test':
            #         bboxes = self.load_bboxes(im_fn)
            #     ann = dict(
            #         filename=im_fn,
            #         width=width,
            #         height=height,
            #         bboxes=bboxes
            #     )
            #     self.anns_list.append(ann)
            #
            # for idx in self.del_idx_list:
            #     im_fns.remove(im_fns[idx])
            # gt_fns.remove(gt_fns[idx])
            for p in processes:
                p.terminate()
                p.join()

            for im_fn, action, angle in self.IGNORE_IM_FNS[dataset_type]:
                if im_fn not in im_fns:
                    continue

                idx = im_fns.index(im_fn)
                assert im_fn == self.anns_list[idx]['filename'], im_fn + ' ' + self.anns_list[idx]['filename']
                if action == 'rotate':
                    ann = self.anns_list[idx]
                    height, width = ann['height'], ann['width']
                    new_bboxes = []
                    for bbox_str in ann['bboxes']:
                        splits = bbox_str.split(',')
                        pts = np.array([int(x) for x in splits[:8]]).reshape(-1, 2)
                        if angle == 90:
                            new_pts = np.array([(width - y, x) for x, y in pts]).flatten().tolist()
                        elif angle == 180:
                            new_pts = np.array([(width - x, height - y) for x, y in pts]).flatten().tolist()
                        else:
                            new_pts = pts.flatten().tolist()
                        new_bbox = ','.join([','.join([str(x) for x in new_pts]), ','.join(splits[8:])])
                        new_bboxes.append(new_bbox)
                    ann['bboxes'] = new_bboxes
                elif action == 'delete':
                    del self.anns_list[idx]
                    del im_fns[idx]
            self.anns_list_tmp = []
            self.anns_list_tmp.extend(self.anns_list)
            self.anns_list = self.anns_list_tmp
            with open(ann_file, 'w', encoding='utf-8') as f:
                json.dump(self.anns_list, f)
        else:
            with open(ann_file, 'r', encoding='utf-8') as f:
                self.anns_list = json.load(f)
        # self.anns_list=mp.Manager().list(self.anns_list)
        # print(type(self.anns_list))
        # print('Load {} images of train dataset -- {}'.format(len(self.anns_list), imgs_dir))
        # gc.disable()
        # processes = []
        # manager = Manager()
        # queue = manager.Queue(maxsize=500)
        # for _ in range(num_process):
        #     p = mp.Process(target=self.single_poly, args=(queue,))
        #     p.start()
        #     processes.append(p)
        #
        # for idx, ann in tqdm(enumerate(self.anns_list)):
        #     if idx==200:
        #         break
        #     queue.put((idx,ann))
        #
        # for p in processes:
        #     p.terminate()
        #     p.join()
        # print(self.anns_list[0])
        # gc.enable()
        return self.anns_list

    def single_poly(self, queue):
        while True:
            if not queue.empty():
                idx, ann = queue.get()
                ann.update(poly_label=[])
                for bbox_str in ann['bboxes']:
                    splits = bbox_str.split(',')
                    poly, script, transcription = splits[:-2], splits[-2], splits[-1]
                    poly = np.array([float(x) for x in poly]).reshape(-1, 2)
                    ann['poly_label'].append(self.BBox_type(poly).gen_poly_label())
                    self.anns_list[idx] = ann.copy()

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None or data['gt_bboxes'].size(0) == 0 or 'fruits_129' in data['img_meta'].data['filename']:
                idx = self._rand_another(idx)
                continue
            return data

    def get_ann_info(self, idx):
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
        for i, bbox_item in enumerate(self.ann_list[idx]['bboxes']):
            poly, script, transcription = bbox_item['bbox'], bbox_item['script'], bbox_item['transcription']
            poly = np.array([float(x) for x in poly]).reshape(-1, 2)
            poly_label = self.BBox_type(poly, num_width_line=self.num_width_line).gen_poly_label()

            if poly_label is None:
                # bbox = (np.min(poly[:, 0]), np.min(poly[:, 1]),
                #         np.max(poly[:, 0]), np.max(poly[:, 1]))
                # gt_bboxes_ignore.append(bbox)
                gt_bboxes_ignore.append(poly)
            elif script == '#':
                # gt_bboxes_ignore.append(poly_label['bbox'])
                gt_bboxes_ignore.append(poly)
            else:
                gt_bboxes.append(poly_label['bbox'])
                gt_labels.append(self.cat2label[script])
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
                gt_transcriptions.append(transcription)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_masks_ann = gt_masks_ann
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_masks_ann = [[[0, 0, 0, 0, 0, 0]]]

        # if gt_bboxes_ignore:
        #     gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        # else:
        #     gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

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
        return ann
