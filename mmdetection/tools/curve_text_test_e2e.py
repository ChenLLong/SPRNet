import argparse
import os
import os.path as osp
import shutil
import tempfile
import numpy as np
from tqdm import tqdm
import time
import subprocess
from time import *
import editdistance
from weighted_editdistance import weighted_edit_distance
import cv2
import mmcv
import pycocotools.mask as maskUtils
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.datasets import *

def load_lexicon(dataset, lexicon_dir, lexicon_type=3):
    '''
    :param dataset:
    :param lexicon_type: 1 for generic; 2 for weak; 3 for strong
    :param weighted_ed:
    :return:
    '''
    if isinstance(dataset, ICDAR2015Dataset):
        if lexicon_type == 1:
            # generic lexicon
            pair_list = open(osp.join(lexicon_dir, 'GenericVocabulary_pair_list.txt'), 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
            lexicon_fid = open(osp.join(lexicon_dir, 'GenericVocabulary_new.txt'), 'r')
            lexicon = []
            for line in lexicon_fid.readlines():
                line = line.strip()
                lexicon.append(line)
        if lexicon_type == 2:
            # weak lexicon
            pair_list = open(osp.join(lexicon_dir, 'ch4_test_vocabulary_pair_list.txt'), 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
            lexicon_fid = open(osp.join(lexicon_dir, 'ch4_test_vocabulary_new.txt'), 'r')
            lexicon = []
            for line in lexicon_fid.readlines():
                line = line.strip()
                lexicon.append(line)
        if lexicon_type == 3:
            pairs_per_img = dict()
            lexicon_per_img = dict()
            for i in range(1,501):
                # weak
                pair_list = open(osp.join(lexicon_dir, 'new_strong_lexicon/pair_voc_img_' + str(i) + '.txt'), 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line = line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word) + 1:]
                    pairs[word] = word_gt
                pairs_per_img['img_' + str(i)] = pairs
                lexicon_fid = open(osp.join(lexicon_dir, 'new_strong_lexicon/new_voc_img_' + str(i) + '.txt'), 'r')
                lexicon = []
                for line in lexicon_fid.readlines():
                    line = line.strip()
                    lexicon.append(line)
                lexicon_per_img['img_' + str(i)] = lexicon
            pairs = pairs_per_img
            lexicon = lexicon_per_img
    elif isinstance(dataset, TotalTextDataset):
        if lexicon_type == 2:
            # weak lexicon
            pair_list = open(osp.join(lexicon_dir, 'weak_voc_pair_list.txt'), 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
            lexicon_fid = open(osp.join(lexicon_dir, 'weak_voc_new.txt'), 'r')
            lexicon = []
            for line in lexicon_fid.readlines():
                line = line.strip()
                lexicon.append(line)
        else:
            raise NotImplementedError
    elif isinstance(dataset, CTW1500Dataset):
        if lexicon_type == 2:
            # weak lexicon
            pair_list = open(osp.join(lexicon_dir, 'weak_voc_pair_list.txt'), 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
            lexicon_fid = open(osp.join(lexicon_dir, 'weak_voc_new.txt'), 'r')
            lexicon = []
            for line in lexicon_fid.readlines():
                line = line.strip()
                lexicon.append(line)
        else:
            raise NotImplementedError
    elif isinstance(dataset, ICDAR2013Dataset):
        if lexicon_type == 1:
            # generic lexicon
            pair_list = open(osp.join(lexicon_dir, 'GenericVocabulary_pair_list.txt'), 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
            lexicon_fid = open(osp.join(lexicon_dir, 'GenericVocabulary_new.txt'), 'r')
            lexicon = []
            for line in lexicon_fid.readlines():
                line = line.strip()
                lexicon.append(line)
        if lexicon_type == 2:
            # weak lexicon
            pair_list = open(osp.join(lexicon_dir, 'ch4_test_vocabulary_pair_list.txt'), 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
            lexicon_fid = open(osp.join(lexicon_dir, 'ch4_test_vocabulary_new.txt'), 'r')
            lexicon = []
            for line in lexicon_fid.readlines():
                line = line.strip()
                lexicon.append(line)
        for i in range(1, 234):
            pairs_per_img = dict()
            lexicon_per_img = dict()
            if lexicon_type == 3:
                # weak
                pair_list = open(osp.join(lexicon_dir, 'new_strong_lexicon/pair_voc_img_' + str(i) + '.txt'), 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line = line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word) + 1:]
                    pairs[word] = word_gt
                pairs_per_img['img_' + str(i)] = pairs
                lexicon_fid = open(osp.join(lexicon_dir, 'new_strong_lexicon/new_voc_img_' + str(i) + '.txt'), 'r')
                lexicon = []
                for line in lexicon_fid.readlines():
                    line = line.strip()
                    lexicon.append(line)
                lexicon_per_img['img_' + str(i)] = lexicon
            pairs = pairs_per_img
            lexicon = lexicon_per_img
    else:
        raise NotImplementedError
    return lexicon, pairs

def lexicon_correct(result, lexicon, pairs, lexicon_type, weighted_ed):
    '''
    :param pred: det_res, reg_pts_res, recog_res, recog_score
    :param lexicon:
    :param pairs:
    :param lexicon_type:
    :return:
    '''
    img_meta = result['img_meta']
    im_fn = osp.basename(img_meta[0]['filename'])
    det_res, reg_pts_res, recog_res, recog_score = result['pred']
    if lexicon_type == 3:
        lexicon = lexicon[osp.splitext(im_fn)[0]]
        pairs = pairs[osp.splitext(im_fn)[0]]
    recog_res_return = []
    for recog_str, rec_score in zip(recog_res, recog_score):
        rec_str = recog_str.upper()
        dist_min = 100
        dist_min_pre = 100
        match_word = ''
        match_dist = 100
        if not weighted_ed:
            for word in lexicon:
                word = word.upper()
                ed = editdistance.eval(rec_str, word)
                length_dist = abs(len(word) - len(rec_str))
                # dist = ed + length_dist
                dist = ed
                if dist < dist_min:
                    dist_min = dist
                    match_word = pairs[word]
                    match_dist = dist
            # return match_word, match_dist
            recog_res_return.append(match_word)
        else:
            small_lexicon_dict = dict()
            for word in lexicon:
                word = word.upper()
                ed = editdistance.eval(rec_str, word)
                small_lexicon_dict[word] = ed
                dist = ed
                if dist < dist_min_pre:
                    dist_min_pre = dist
            small_lexicon = []
            for word in small_lexicon_dict:
                if small_lexicon_dict[word] <= dist_min_pre + 2:
                    small_lexicon.append(word)

            for word in small_lexicon:
                word = word.upper()
                ed = weighted_edit_distance(rec_str, word, rec_score)
                dist = ed
                if dist < dist_min:
                    dist_min = dist
                    match_word = pairs[word]
                    match_dist = dist
            # return match_word, match_dist
            recog_res_return.append(match_word)
    result['pred'] = det_res, reg_pts_res, recog_res_return, recog_score
    return result


def results2txt(dataset, outputs, output_dir, eval_det=True):
    os.makedirs(output_dir, exist_ok=True)
    if eval_det:
        det_output_dir = os.path.join(output_dir, 'detection_results')
        os.makedirs(det_output_dir, exist_ok=True)
    outputs = tqdm(outputs) if len(outputs) > 1 else outputs
    for output in outputs:
        img_meta = output['img_meta']
        det_res, reg_pts_res, recog_res, recog_score = output['pred']
        im_fn = osp.basename(img_meta[0]['filename'])

        if isinstance(dataset, ICDAR2015Dataset):
            res_txt_fn = ('res_' + osp.splitext(im_fn)[0] + '.txt')
            with open(osp.join(output_dir, res_txt_fn), 'w') as fw:
                for pts, tsp in zip(reg_pts_res, recog_res):
                    bbox = cv2.boxPoints(cv2.minAreaRect(pts)).flatten()
                    # bbox=pts.flatten()
                    fw.write('{}\n'.format(','.join([str(int(x)) for x in bbox]+[tsp])))
            if eval_det:
                res_txt_fn = ('res_' + osp.splitext(im_fn)[0] + '.txt')
                with open(osp.join(det_output_dir, res_txt_fn), 'w') as fw:
                    for pts in reg_pts_res:
                        bbox = cv2.boxPoints(cv2.minAreaRect(pts)).flatten()
                        # bbox = pts.flatten()
                        fw.write('{}\n'.format(','.join([str(int(x)) for x in bbox])))
                        # fw.write('{}\n'.format(','.join([str(int(x)) for x in bbox])))
        elif isinstance(dataset, MLT2017Dataset):
            res_txt_fn = ('res_' + osp.splitext(im_fn)[0] + '.txt').replace('ts_', '')
            with open(osp.join(output_dir, res_txt_fn), 'w') as fw:
                for pts in reg_pts_res:
                    bbox = cv2.boxPoints(cv2.minAreaRect(pts)).flatten()
                    fw.write('{},1.0,{}\n'.format(','.join([str(int(x)) for x in bbox], tsp))) #TODO AP
            if eval_det:
                res_txt_fn = ('res_' + osp.splitext(im_fn)[0] + '.txt').replace('ts_', '')
                with open(osp.join(output_dir, res_txt_fn), 'w') as fw:
                    for pts in reg_pts_res:
                        bbox = cv2.boxPoints(cv2.minAreaRect(pts)).flatten()
                        fw.write('{},1.0\n'.format(','.join([str(int(x)) for x in bbox])))
        elif isinstance(dataset, TotalTextDataset):
            res_txt_fn = ('res_' + osp.splitext(im_fn)[0] + '.txt')
            with open(osp.join(output_dir, res_txt_fn), 'w') as fw:
                for pts, tsp in zip(reg_pts_res, recog_res):
                    fw.write('{}\n'.format(','.join([str(int(x)) for x in pts.flatten()]+[tsp])))
            if eval_det:
                res_txt_fn = osp.splitext(im_fn)[0] + '.mat'
                with open(osp.join(det_output_dir, res_txt_fn), 'w') as fw:
                    for bbox in reg_pts_res:
                        fw.write('{}\n'.format(','.join([str(int(x)) for x in bbox[:, ::-1].flatten()])))
        elif isinstance(dataset, CTW1500Dataset):
            res_txt_fn = ('res_' + osp.splitext(im_fn)[0] + '.txt')
            with open(osp.join(output_dir, res_txt_fn), 'w') as fw:
                for pts, tsp in zip(reg_pts_res, recog_res):
                    fw.write('{}\n'.format(','.join([str(int(x)) for x in pts.flatten()]+[tsp])))
            if eval_det:
                res_txt_fn = osp.splitext(im_fn)[0] + '.txt'
                with open(osp.join(det_output_dir, res_txt_fn), 'w') as fw:
                    for bbox in reg_pts_res:
                        fw.write('{}\n'.format(','.join([str(int(x)) for x in bbox.flatten()])))
        else:
            raise NotImplementedError


def visual_results(dataset, outputs, visual_dir):
    os.makedirs(visual_dir, exist_ok=True)
    outputs = tqdm(outputs) if len(outputs) > 1 else outputs
    for output in outputs:
        try:
            img_meta = output['img_meta']
            det_res, reg_pts_res, tsps, score = output['pred']
            im_fn = osp.basename(img_meta[0]['filename'])
            im = cv2.imread(img_meta[0]['filename'])

            if isinstance(dataset, (ICDAR2015Dataset, MLT2017Dataset, TotalTextDataset)):
                res_txt_fn = ('res_' + osp.splitext(im_fn)[0] + '.txt').replace('ts_', '')
            elif isinstance(dataset, CTW1500Dataset):
                res_txt_fn = osp.splitext(im_fn)[0] + '.txt'
            else:
                raise NotImplementedError

            with open(osp.join(visual_dir, res_txt_fn), 'r') as fr:
                for bbox, line, tsp in zip(det_res[0], fr.readlines(), tsps):
                    x1, y1, x2, y2, score = bbox
                    if isinstance(dataset, MLT2017Dataset):
                        pts = np.array([int(x) for x in line.split(',')[:-1]]).reshape(-1, 2)
                    else:
                        pts = np.array([int(x) for x in line.split(',')[:-1]]).reshape(-1, 2)
                    # cv2.rectangle(im, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
                    cv2.polylines(im, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    height, width, _ = im.shape
                    scale = 1.0
                    x, y = np.min(pts[:, 0]), np.min(pts[:, 1])
                    w, h = int(width / 100 * scale * len(tsp)), int(max(height, width) / 66 * scale)
                    cv2.rectangle(im, (x, y - h), (x + w, y), (255, 255, 255), -1)
                    cv2.putText(im, tsp, (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, width / 2000 * scale, (0, 0, 0), 1)
            cv2.imwrite(osp.join(visual_dir, im_fn), im)
        except Exception as e:
            print(img_meta[0]['filename'])
            import traceback
            traceback.print_exc()


def eval_results(dataset, output_dir, eval_dir, ckpt_path, lexicon_type, use_lexicon, eval_det=True, fps_results=0.0):

    version = osp.basename(output_dir)
    cmd_strs = []
    if isinstance(dataset, ICDAR2015Dataset):
        cmd_strs.append('cd {}; zip -qj {}.zip *.txt'.format(output_dir, version))
        cmd_strs.append('python3 {}/script.py -g={}/gt.zip -s={}'.format(eval_dir, eval_dir,
                                                                         osp.join(output_dir, version + '.zip')))
    elif isinstance(dataset, MLT2017Dataset):
        cmd_strs.append('cd {}; zip -qj {}.zip *.txt'.format(output_dir, version))
    elif isinstance(dataset, TotalTextDataset):
        cmd_strs.append('cd {}; zip -qj {}.zip *.txt'.format(output_dir, version))
        cmd_strs.append('python3 {}/script.py -g={}/gt.zip -s={}'.format(eval_dir, eval_dir,
                                                                         osp.join(output_dir, version + '.zip')))
    elif isinstance(dataset, CTW1500Dataset):
        cmd_strs.append('cd {}; zip -qj {}.zip *.txt'.format(output_dir, version))
        cmd_strs.append('python3 {}/script.py -g={}/gt.zip -s={}'.format(eval_dir, eval_dir,
                                                                         osp.join(output_dir, version + '.zip')))
    else:
        raise NotImplementedError

    for cmd_str in cmd_strs:
        stdout_str = subprocess.getoutput(cmd_str)

    # eval detection
    if eval_det:
        det_output_dir = os.path.join(output_dir, 'detection_results')
        det_version = 'det_' + version
        det_eval_dir = os.path.dirname(eval_dir).replace('_e2e', '')
        cmd_strs = []
        if isinstance(dataset, ICDAR2015Dataset):
            cmd_strs.append('cd {}; zip -qj {}.zip *.txt'.format(det_output_dir, det_version))
            cmd_strs.append('python3 {}/script.py -g={}/gt.zip -s={}'.format(det_eval_dir, det_eval_dir,
                                                                             osp.join(det_output_dir, det_version + '.zip')))
        elif isinstance(dataset, MLT2017Dataset):
            cmd_strs.append('cd {}; zip -qj {}.zip *.txt'.format(det_output_dir, det_version))
        elif isinstance(dataset, TotalTextDataset):
            cmd_strs.append(
                'python3 {}/eval_totaltext.py {} {}'.format(det_eval_dir, osp.dirname(dataset.ann_file), det_output_dir))
        elif isinstance(dataset, CTW1500Dataset):
            cmd_strs.append('python3 {}/eval_ctw1500.py {} {}'.format(det_eval_dir, osp.dirname(dataset.ann_file), det_output_dir))
        else:
            raise NotImplementedError

        for cmd_str in cmd_strs:
            eval_det_stdout_str = subprocess.getoutput(cmd_str)

    log_fp = osp.join(output_dir, version + '.log')
    import time
    with open(log_fp, 'a') as log_fw:
        if use_lexicon:
            log_test_type = 'use lexicon type {}'.format(lexicon_type)
        else:
            log_test_type = 'not use lexicon'
        log_fw.write(ckpt_path + '\t' + log_test_type + '\n')
        log_fw.write('\t' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\t' + 'e2e results' + '\t' + stdout_str + '\n')
        log_fw.write('\t' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\t' + 'detection results' + '\t' + eval_det_stdout_str + '\n')
        log_fw.write('\t' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\t' + 'fps results' + '\t' + str(fps_results) + '\n\n')
    print('current test preformance: \n e2e -- {} \n det -- {} \n fps -- {}'.format(stdout_str, eval_det_stdout_str, fps_results))


def single_gpu_test(model, data_loader, output_dir, lexicon_dir, lexicon_type, use_lexicon, weighted_ed=True, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if use_lexicon:
        lexicon, pairs = load_lexicon(dataset, lexicon_dir, lexicon_type)
    run_time=0
    for i, data in enumerate(data_loader):
        begin_time=time()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        end_time=time()
        run_time += end_time-begin_time
        result = {
            'img_meta': data['img_meta'][0].data[0],
            'pred': result
        }

        if use_lexicon:
            result = lexicon_correct(result, lexicon, pairs, lexicon_type, weighted_ed)

        results.append(result)

        results2txt(dataset, [result], output_dir)

        if show:
            # model.module.show_result(data, result)
            visual_results(dataset, [result], output_dir)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    fps_results = len(dataset) / run_time
    print('\n total {} fps'.format(fps_results))
    return results, fps_results


def multi_gpu_test(model, data_loader, output_dir, lexicon_dir, lexicon_type, use_lexicon, weighted_ed=True, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    fps_results = 0.0
    if rank == 0:
        num_data_process = 0
        prog_bar = mmcv.ProgressBar(len(dataset))
    if use_lexicon:
        lexicon, pairs = load_lexicon(dataset, lexicon_dir, lexicon_type)
    run_time=0
    for i, data in enumerate(data_loader):
        begin_time = time()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        end_time = time()
        run_time += end_time-begin_time
        result = {
            'img_meta': data['img_meta'][0].data[0],
            'pred': result
        }

        if use_lexicon:
            result = lexicon_correct(result, lexicon, pairs, lexicon_type, weighted_ed)

        results.append(result)

        results2txt(dataset, [result], output_dir)

        if show:
            # model.module.show_result(data, result)
            visual_results(dataset, [result], output_dir)

        if rank == 0:
            num_data_process = i
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    if rank == 0:
        fps_results = (num_data_process + 1) / run_time
        print('\n total {} fps'.format(fps_results))
    # collect results from all ranks
    # results = collect_results(results, len(dataset), tmpdir)

    return results, fps_results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--version', type=str, default='', help='version of current experiment')
    parser.add_argument('--scratch', action='store_true', help='whether to train from scratch or not')
    parser.add_argument('--datasets', type=str, default='', help='dataset type')
    parser.add_argument('--multiscale', action='store_true', help='whether to test with multi-scale or not')
    parser.add_argument('--use_lexicon', action='store_true', help='whether to use lexicon or not')
    parser.add_argument('--lexicon_type', choices=[1, 2, 3], type=int, default=1, help='1 for generic; 2 for weak; 3 for strong')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    for test_data_cfg in cfg.data.test:
        test_data_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    #############################################################
    if len(args.version.split('_')) == 4:
        degree, control_point, width_line, attention = args.version.split('_')
        cfg.model.mask_head.b_spline_degree = int(degree[1:])
        cfg.model.mask_head.num_control_point = int(control_point[1:])
        cfg.model.mask_head.num_width_line = int(width_line[1:])
        if attention[1] == '1':
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
        elif attention[1] == '0':
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
                roi_layer=dict(type='RoIAlign', out_size=(16, 64), sample_num=2),
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
        cfg.data.test = []
    data_root = cfg.data_root
    test_pipeline = cfg.test_pipeline
    for ds in datasets:
        if ds == 'ic15':
            cfg.data.test.append(
                dict(
                    type='ICDAR2015Dataset',
                    ann_file=data_root + 'icdar_2015/task1/test/gts/test_gts.json',
                    img_prefix=data_root + 'icdar_2015/task1/test/images',
                    pipeline=test_pipeline,test_mode=True),
            )
            # img_scale = (1920, 1080) if not args.multiscale else [(1280, 720), (1920, 1080), (2560, 1440)]
            img_scale = (2200, 1320) if not args.multiscale else [(1200, 720), (2200, 1320), (3200, 1920)]
        elif ds == '17mlt':
            cfg.data.test.append(
                dict(
                    type='MLT2017Dataset',
                    ann_file=data_root + 'icdar_2017_MLT/task1/test/gts/test_gts.json',
                    img_prefix=data_root + 'icdar_2017_MLT/task1/test/images',
                    pipeline=test_pipeline,test_mode=True)
            )
            img_scale = (1920, 1080) if not args.multiscale else [(1280, 720), (1920, 1080), (2560, 1440)]
        elif ds == 'totaltext':
            cfg.data.test.append(
                dict(
                    type='TotalTextDataset',
                    ann_file=data_root + 'total_text/test/mat_gts/polygon/test_gts.json',
                    img_prefix=data_root + 'total_text/test/images',
                    pipeline=test_pipeline,test_mode=True),
            )
            img_scale = (1200, 720) if not args.multiscale else [(1333, 800), (1200, 720), (1066, 640)]
        elif ds == 'ctw1500':
            cfg.data.test.append(
                dict(
                    type='CTW1500Dataset',
                    ann_file=data_root + 'ctw1500/test/gts/test_gts.json',
                    img_prefix=data_root + 'ctw1500/test/images',
                    pipeline=test_pipeline,test_mode=True),
            )
            img_scale = (1200, 720) if not args.multiscale else [(1333, 800), (1200, 720), (1066, 640)]

    if args.multiscale:
        cfg.test_cfg.rcnn['score_thr'] = 0.65
    else:
        cfg.test_cfg.rcnn['score_thr'] = 0.70
    for test_ds in cfg.data.test:
        test_ds['pipeline'] = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_scale,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='TextRandomFlip'),
                    dict(type='Normalize', **cfg.img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
    # rank, _ = get_dist_info()
    # if rank==0:
    #     for key in cfg:
    #         print('{}:{}'.format(key,cfg[key]))

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    datasets = [build_dataset(test_data_cfg) for test_data_cfg in cfg.data.test]
    data_loaders = [build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
        for dataset in datasets]

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # if 'CLASSES' in checkpoint['meta']:
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = datasets[0].CLASSES

    for data_loader in data_loaders:
        dataset = data_loader.dataset

        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if isinstance(dataset, ICDAR2015Dataset):
            dataset_name = 'icdar_2015'
        elif isinstance(dataset, MLT2017Dataset):
            dataset_name = 'icdar_2017_MLT'
        elif isinstance(dataset, TotalTextDataset):
            dataset_name = 'total_text'
        elif isinstance(dataset, CTW1500Dataset):
            dataset_name = 'ctw1500'
        else:
            dataset_name = 'temp'
        output_dir = osp.join(cfg.output_dir, dataset_name, 'results', osp.basename(osp.dirname(args.checkpoint)))
        lexicon_dir = osp.join(cfg.eval_dir, 'lexicons', dataset_name)
        visual_dir = output_dir + '_vis'
        eval_dir = osp.join(cfg.eval_dir, dataset_name, 'e2e')
        out_fp = osp.join(output_dir, args.out)

        rank, _ = get_dist_info()
        if args.out:
            eval_types = args.eval
            if eval_types == ['proposal_fast']:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                eval_results(dataset, output_dir, eval_dir, args.checkpoint, args.lexicon_type, args.use_lexicon, eval_det=True, fps_results=0.0)
            else:
                if not distributed:
                    model = MMDataParallel(model, device_ids=[0])
                    # outputs = single_gpu_test(model, data_loader, args.show)
                    # def single_gpu_test(model, data_loader, output_dir, lexicon_dir, lexicon_type, use_lexicon, weighted_ed=True, show=False):
                    outputs, fps_results = single_gpu_test(model, data_loader, output_dir, lexicon_dir, args.lexicon_type, args.use_lexicon, weighted_ed=False, show=args.show)
                else:
                    model = MMDistributedDataParallel(model.cuda())
                    outputs, fps_results = multi_gpu_test(model, data_loader, output_dir, lexicon_dir, args.lexicon_type, args.use_lexicon, weighted_ed=False, show=args.show)
                if rank == 0:
                    print('\nwriting results to {}'.format(out_fp))
                    os.makedirs(osp.dirname(out_fp), exist_ok=True)
                    mmcv.dump(outputs, out_fp)

                    print('Starting evaluate {}'.format(' and '.join(eval_types)))
                    if not isinstance(outputs[0]['pred'], dict):
                        # results2txt(dataset, outputs, output_dir)
                        # if args.show:
                        #     visual_results(dataset, outputs, visual_dir)
                        eval_results(dataset, output_dir, eval_dir, args.checkpoint, args.lexicon_type, args.use_lexicon, eval_det=True, fps_results=fps_results)


if __name__ == '__main__':
    main()
