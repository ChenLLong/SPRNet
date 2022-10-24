from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset, build_dataset_ratio
from mmdet.models import build_detector
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet.datasets.textdatasets.polybbox_v1 import PolyBBox as PolyBBox_divide
from mmdet.datasets.textdatasets.polybbox import PolyBBox as PolyBBox_ori
from mmdet.datasets.textdatasets.polybbox_Q import PolyBBox as PolyBBox_q
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    parser.add_argument('--version', type=str, default='', help='version of current experiment')
    parser.add_argument('--scratch', action='store_true', help='whether to train from scratch or not')
    parser.add_argument('--datasets', type=str, default='', help='dataset type')
    parser.add_argument('--multiscale', action='store_true', help='whether to train with multi-scale or not')
    parser.add_argument('--load_from', help='load pre-train checkpoint or None')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn|BaiduCTC')
    parser.add_argument(
            '--mode',
            choices=['pretrain', 'finetune'],
            default='pretrain',
            help='job launcher')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    print(args)
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    #####################################################
    if len(args.version.split('_')) == 4:
        degree, control_point, width_line, attention= args.version.split('_')
        cfg.model.mask_head.b_spline_degree = int(degree[1:])
        cfg.model.mask_head.num_control_point = int(control_point[1:])
        cfg.model.mask_head.num_width_line = int(width_line[1:])
        if attention[1]=='1':
            cfg.model.bbox_roi_extractor = dict(
                type='MultiRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
            cfg.model.mask_roi_extractor = dict(
                type='MultiRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=16, sample_num=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
                no_spatial=False)
            cfg.model.recognition_roi_extractor = dict(
                type='MultiRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=16, sample_num=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
                no_spatial=False)
        elif attention[1]=='0':
            cfg.model.bbox_roi_extractor = dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
            cfg.model.mask_roi_extractor = dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=16, sample_num=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
            cfg.model.recognition_roi_extractor = dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=(16,64), sample_num=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.scratch:
        cfg.model.pretrained = None

    datasets = args.datasets.split('_')
    if len(datasets) != 0:
        cfg.data.train = []
        img_scale_list=[]
    data_root = cfg.data_root
    train_pipeline = cfg.train_pipeline
    for ds in datasets:
        if ds == 'ic15':
            cfg.data.train.append(
                dict(
                    type='ICDAR2015Dataset',
                    ann_file=data_root + 'icdar_2015/task1/train/gts/train_gts.json',
                    img_prefix=data_root + 'icdar_2015/task1/train/images/',
                    pipeline=train_pipeline, regen_ann=False, num_width_line = cfg.model.mask_head.num_width_line, BBox_type=PolyBBox_divide, ratio=3),
            )
            img_scale = (1333,800) if not args.multiscale else [(1333,800), (1280, 720), (1920,1080)]
            img_scale_list.append(img_scale)
        elif ds == '17mlt':
            cfg.data.train.append(
                dict(
                    type='MLT2017Dataset',
                    ann_file=[data_root + 'icdar_2017_MLT/task1/train/gts/train_gts.json',
                              data_root + 'icdar_2017_MLT/task1/val/gts/val_gts.json'],
                    img_prefix=[data_root + 'icdar_2017_MLT/task1/train/images', data_root + 'icdar_2017_MLT/task1/val/images'],
                    pipeline=train_pipeline, regen_ann=False, num_width_line = cfg.model.mask_head.num_width_line, BBox_type=PolyBBox_divide)
            )
            img_scale = (1333,800) if not args.multiscale else [(1333,800), (1280, 720), (1920,1080)]
            img_scale_list.append(img_scale)
        elif ds == 'totaltext':
            cfg.data.train.append(
                dict(
                    type='TotalTextDataset',
                    ann_file=data_root + 'total_text/train/new_mat_gts/polygon/train_gts.json',
                    img_prefix=data_root + 'total_text/train/images',
                    pipeline=train_pipeline, regen_ann=False, num_width_line = cfg.model.mask_head.num_width_line, BBox_type=PolyBBox_divide, ratio=3),
            )
            img_scale = (1200, 720) if not args.multiscale else [(1333, 800), (1200, 720), (1066, 640), (840,540)]
            img_scale_list.append(img_scale)
        elif ds == 'ctw1500':
            cfg.data.train.append(
                dict(
                    type='CTW1500Dataset',
                    ann_file=data_root + 'ctw1500/train/gts_e2e_xml/train_gts.json',
                    img_prefix=data_root + 'ctw1500/train/images',
                    pipeline=train_pipeline,regen_ann=True, num_width_line = cfg.model.mask_head.num_width_line, BBox_type=PolyBBox_divide, ratio=3),
            )
            img_scale = (1200, 720) if not args.multiscale else [(1333, 800), (1200, 720), (1066, 640)]
            img_scale_list.append(img_scale)
        elif ds == 'synthtext':
            cfg.data.train.append(
                dict(
                    type='SynthTextDataset',
                    ann_file=data_root + 'synthtext/train/gts/train_gts.json',
                    img_prefix=data_root + 'synthtext/train/images',
                    pipeline=train_pipeline, regen_ann=False, num_width_line=cfg.model.mask_head.num_width_line, BBox_type=PolyBBox_divide, ratio=3),
            )
            img_scale = (1200, 720) if not args.multiscale else [(1333, 800), (1200, 720), (1066, 640), (840, 540)]
            img_scale_list.append(img_scale)
        elif ds == 'ic13':
            cfg.data.train.append(
                dict(
                    type='ICDAR2013Dataset',
                    ann_file=data_root + 'icdar_2013/task1/train/gts/train_gts.json',
                    img_prefix=data_root + 'icdar_2013/task1/train/images/',
                    pipeline=train_pipeline, regen_ann=False, num_width_line=cfg.model.mask_head.num_width_line, BBox_type=PolyBBox_divide, ratio=1),
            )
            img_scale = (1333, 800) if not args.multiscale else [(1333, 800), (1280, 720), (840, 540)]
            img_scale_list.append(img_scale)
        elif ds == 'cocotext':
            cfg.data.train.append(
                dict(
                    type='COCOTextDataset',
                    ann_file=[data_root + 'coco_text/train/gts/train_gts.json',
                              data_root + 'coco_text/val/gts/train_gts.json'],
                    img_prefix=[data_root + 'coco_text/train/images/',
                                data_root + 'coco_text/val/images/'],
                    pipeline=train_pipeline, regen_ann=False, num_width_line=cfg.model.mask_head.num_width_line, BBox_type=PolyBBox_divide, ratio=2),
            )
            img_scale = (1333, 800) if not args.multiscale else [(1333, 800), (1280, 720), (840, 540)]
            img_scale_list.append(img_scale)
        elif ds == 'scut':
            cfg.data.train.append(
                dict(
                    type='SCUTDataset',
                    ann_file=data_root + 'scut/train/gts/train_gts.json',
                    img_prefix=data_root + 'scut/train/images/',
                    pipeline=train_pipeline, regen_ann=False, num_width_line=cfg.model.mask_head.num_width_line, BBox_type=PolyBBox_divide, ratio=1),
            )
            img_scale = (1333, 800) if not args.multiscale else [(1333, 800), (1280, 720), (840, 540)]
            img_scale_list.append(img_scale)

    for i, train_ds in enumerate(cfg.data.train):
        if isinstance(img_scale_list, list):
            resize = dict(type='TextResize', img_scale=img_scale_list[i], multiscale_mode='value', keep_ratio=True)
        else:
            resize=dict(type='TextResize', img_scale=img_scale_list[i], keep_ratio=True)
        if args.mode == 'pretrain' and train_ds['type'] in ['ICDAR2015Dataset']:
            train_ds['pipeline'] = [
                dict(type='LoadTextImageFromFile', to_float32=True),
                dict(type='LoadTextAnnotations', with_bbox=True, with_mask=True),
                dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
                resize,
                dict(type='TextRandomFlip', flip_ratio=0.),
                dict(type='TextMinIoURandomCrop', crop_size=960),
                dict(type='TextRandomRotate', rotate_ratio=0.5, BBox_type=PolyBBox_divide, max_rotate_angle=45, num_width_line=cfg.model.mask_head.num_width_line),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='TextDefaultFormatBundle', character=cfg.character, recognition_predition_type=args.Prediction),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_poly_pts',
                                           'gt_center_line', 'gt_width_line', 'gt_line_angle', 'gt_line_length', 'gt_line_x', 'gt_line_y',
                                           'gt_direction', 'gt_transcriptions', 'gt_decoders', 'gt_text_lengths'])
            ]
        else:
            train_ds['pipeline'] = [
                dict(type='LoadTextImageFromFile', to_float32=True),
                dict(type='LoadTextAnnotations', with_bbox=True, with_mask=True),
                dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
                resize,
                dict(type='TextRandomFlip', flip_ratio=0.),
                dict(type='TextRandomRotate', rotate_ratio=0.5, BBox_type=PolyBBox_divide, max_rotate_angle=45, num_width_line=cfg.model.mask_head.num_width_line),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='TextDefaultFormatBundle', character=cfg.character,  recognition_predition_type=args.Prediction),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_poly_pts',
                                           'gt_center_line', 'gt_width_line', 'gt_line_angle', 'gt_line_length', 'gt_line_x', 'gt_line_y',
                                           'gt_direction', 'gt_transcriptions', 'gt_decoders', 'gt_text_lengths'])
            ]

    if args.load_from is None:
        cfg.load_from = None
    else:
        cfg.load_from = args.load_from


    if args.mode == 'pretrain' and 'synthtext' not in args.datasets:
        cfg.total_epochs=80
        cfg.lr_config['step']=[70]
        cfg.optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
        cfg.exchang_anchor_t_config = dict(exchange_t_epoch=1)
        cfg.use_poly_loss_epoch_config = dict(use_poly_loss_epoch=0)

    if args.mode == 'pretrain' and 'synthtext' in args.datasets:
        cfg.total_epochs=1
        cfg.lr_config = dict(
            by_epoch=False,
            policy='step',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=1.0 / 3,
            step=[70000])
        cfg.optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
        cfg.every_n_checkpoint_config = dict(interval=5000)
        cfg.exchang_anchor_t_config = dict(exchange_t_epoch=1)
        cfg.use_poly_loss_epoch_config = dict(use_poly_loss_epoch=0)
    elif args.mode == 'finetune' and args.datasets == 'ic15':
        cfg.data.imgs_per_gpu = 3
        cfg.total_epochs=50
        cfg.lr_config['step']=[40]
        cfg.optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
        cfg.exchang_anchor_t_config = dict(exchange_t_epoch=0)
        cfg.use_poly_loss_epoch_config = dict(use_poly_loss_epoch=0)
    elif args.mode == 'finetune' and args.datasets == 'totaltext':
        cfg.total_epochs=140
        cfg.lr_config['step']=[50, 100]
        cfg.optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
        cfg.exchang_anchor_t_config = dict(exchange_t_epoch=0)
        cfg.use_poly_loss_epoch_config = dict(use_poly_loss_epoch=0)
        cfg.checkpoint_config = dict(interval=1)
    elif args.mode == 'finetune' and args.datasets == 'ctw1500':
        cfg.total_epochs=70
        cfg.lr_config['step']=[40,60]
        cfg.optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
        cfg.exchang_anchor_t_config = dict(exchange_t_epoch=0)
        cfg.use_poly_loss_epoch_config = dict(use_poly_loss_epoch=0)
    elif args.mode == 'finetune' and args.datasets == '17mlt':
        cfg.total_epochs=10
        cfg.lr_config['step']=[8]
        cfg.optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
        cfg.exchang_anchor_t_config = dict(exchange_t_epoch=0)
        cfg.use_poly_loss_epoch_config = dict(use_poly_loss_epoch=0)



    rank, _ = get_dist_info()
    if rank==0:
        for key in cfg:
            print('{}:{}'.format(key,cfg[key]))

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    if isinstance(cfg.data.train, list) and 'ratio' in cfg.data.train[0]:
        print('bulid dataset ration ...')
        datasets = [build_dataset_ratio(cfg.data.train)]
    else:
        datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
