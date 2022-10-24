import math
import random
import time
import numpy as np
from ..registry import RECOGNIZERS
import torch
# from ..utils.chars import char2num, num2char
from torch import nn
from torch.nn import functional as F
from mmdet.core import recognition_target
from mmdet.datasets.pipelines.recognition_label_convert import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
import itertools

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")


def reduce_mul(l):
    out = 1.0
    for x in l:
        out *= x
    return out


def check_all_done(seqs):
    for seq in seqs:
        if not seq[-1]:
            return False
    return True


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


# output_ctrl_pts are specified, according to our task.
def build_output_control_points(num_control_points, margins):
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = num_control_points // 2
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    # ctrl_pts_top = ctrl_pts_top[1:-1,:]
    # ctrl_pts_bottom = ctrl_pts_bottom[1:-1,:]
    output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
    output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
    #   print(output_ctrl_pts)
    return output_ctrl_pts


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size()).fill_(1)
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

@RECOGNIZERS.register_module
class SRRHead(nn.Module):
    def __init__(self, dim_in, Rect_ON=False, num_of_slices=6, num_char=38, teacher_force_ratio=1.0, character=[], transformation_type='pt', targetH=8, targetW=32):
        super(SRRHead, self).__init__()
        self.Rect_ON = Rect_ON
        self.targetH = targetH
        self.targetW = targetW

        # parameters of the recognition models
        self.NUM_CHAR = num_char
        self.TEACHER_FORCE_RATIO = teacher_force_ratio

        assert transformation_type in ['pt', 'tps'], 'transformation type error ...'
        self.transformation_type = transformation_type
        self.rescale = nn.Upsample(size=(8, 32), mode='bilinear', align_corners=False)
        if self.Rect_ON:
            if self.transformation_type == 'pt':
                self.num_of_slices = num_of_slices
                # get the target control points
                target_ctp_x = torch.arange(self.num_of_slices+1, device=gpu_device, dtype=torch.float) / self.num_of_slices
                target_ctp_bottom_y = torch.zeros(self.num_of_slices+1, device=gpu_device, dtype=torch.float)
                target_ctp_top_y = torch.ones(self.num_of_slices+1, device=gpu_device, dtype=torch.float)
                self.target_ctps = torch.stack([target_ctp_x, target_ctp_bottom_y, target_ctp_x, target_ctp_top_y], -1) # (num_of_slices+1) * 4
                # get the normal grid
                h_list = np.arange(self.targetH) / (self.targetH - 1)
                w_list = np.arange(self.targetW) / (self.targetW - 1)
                constant = np.ones((self.targetW, self.targetH, 1))
                grid = np.meshgrid(
                    w_list,
                    h_list,
                    indexing='ij'
                )
                grid = np.stack(grid, axis=-1)
                grid = np.concatenate((grid, constant), axis=-1)
                grid = np.transpose(grid, (2, 1, 0))
                grid = np.expand_dims(grid, 0)
                grid = torch.tensor(grid.tolist(), device=gpu_device)
                self.target_grid = grid
            elif self.transformation_type == 'tps':
                self.num_control_points = (num_of_slices + 1) * 2
                self.margins = (0., 0.)
                target_control_points = build_output_control_points(self.num_control_points, self.margins)
                N = self.num_control_points

                forward_kernel = torch.zeros(N + 3, N + 3)
                target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
                forward_kernel[:N, :N].copy_(target_control_partial_repr)
                forward_kernel[:N, -3].fill_(1)
                forward_kernel[-3, :N].fill_(1)
                forward_kernel[:N, -2:].copy_(target_control_points)
                forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

                inverse_kernel = torch.inverse(forward_kernel)
                HW = self.targetH * self.targetW
                target_coordinate = list(itertools.product(range(self.targetH), range(self.targetW)))
                target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
                Y, X = target_coordinate.split(1, dim=1)
                Y = Y / (self.targetH - 1)
                X = X / (self.targetW - 1)
                target_coordinate = torch.cat((X, Y), dim=1)  # convert from (y, x) to (x, y)
                target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
                target_coordinate_repr = torch.cat([
                    target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
                ], dim=1)

                self.register_buffer('inverse_kernel', inverse_kernel)
                self.register_buffer('padding_matrix', torch.zeros(3, 2))
                self.register_buffer('target_coordinate_repr', target_coordinate_repr)
                self.register_buffer('target_control_points', target_control_points)
            else:
                raise ValueError

        self.seq_encoder = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(dim_in, dim_in, 3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(dim_in, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AvgPool2d((2, 1), (2, 1))
        )

        # self.biLSTM = nn.LSTM(256, 128, bidirectional=True, num_layers=2, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.seq_decoder = BahdanauAttnDecoderRNN(
            256, self.NUM_CHAR, self.NUM_CHAR, n_layers=1, dropout_p=0.1, onehot_size=(8, 32)
        )
        self.criterion_seq_decoder = nn.NLLLoss(ignore_index=0, reduction="none")
        self.character = character
        self.convert = AttnLabelConverter(self.character)

    def init_weights(self):
        # for name, param in self.named_parameters():
        #     print(name)
        #     if "bias" in name:
        #         nn.init.constant_(param, 0)
        #     elif "weight" in name:
        #         # Caffe2 implementation uses MSRAFill, which in fact
        #         # corresponds to kaiming_normal_ in PyTorch
        #         nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                # print(m)
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRUCell) or isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    # print(name)
                    if "bias" in name:
                        nn.init.constant_(param, 0)
                    elif "weight" in name:
                        # Caffe2 implementation uses MSRAFill, which in fact
                        # corresponds to kaiming_normal_ in PyTorch
                        nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # else:
            #     print(m)

    def get_target(self, sampling_results, gt_recognitions, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        recognition_targets = recognition_target(pos_proposals, pos_assigned_gt_inds,
                                                     gt_recognitions, rcnn_train_cfg)
        return recognition_targets

    def forward(self, x, control_points, decoder_targets=None, word_targets=None, use_beam_search=True):
        # encode the feature and squeeze the height ot 1
        # start = time.time()
        if self.Rect_ON:
            if self.transformation_type == 'pt':
                x = self.rectification_pt_2(x, control_points)
            elif self.transformation_type == 'tps':
                x = self.rectification_tps(x, control_points)
        if self.targetH != 8:
            x = self.rescale(x)
        cnn_feat = self.seq_encoder(x).squeeze(2)
        cnn_feat = cnn_feat.transpose(2, 1)  # B * T * C
        # rnn_feat, _ = self.biLSTM(cnn_feat)
        rnn_feat = self.transformer(cnn_feat.transpose(0, 1))
        seq_decoder_input_reshape = rnn_feat#.transpose(1, 0)

        if self.training:
            decoder_input = word_targets[:, 0]
            decoder_hidden = torch.zeros(
                (seq_decoder_input_reshape.size(1), 256), device=gpu_device
            )
            use_teacher_forcing = (
                True
                if random.random() < self.TEACHER_FORCE_RATIO
                else False
            )
            target_length = decoder_targets.size(1)
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(1, target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                        decoder_input, decoder_hidden, seq_decoder_input_reshape
                    )
                    if di == 1:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    decoder_input = decoder_targets[:, di]  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(1, target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                        decoder_input, decoder_hidden, seq_decoder_input_reshape
                    )
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(1).detach()  # detach from history as input
                    if di == 1:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
            loss = dict()
            loss_seq_decoder = loss_seq_decoder.sum() / loss_seq_decoder.size(0)
            loss_seq_decoder = 0.2 * loss_seq_decoder
            loss['loss_rec'] = loss_seq_decoder
            # end = time.time()
            # print(end-start)
            return loss
        else:
            words = []
            decoded_scores = []
            detailed_decoded_scores = []
            # real_length = 0
            if use_beam_search:
                for batch_index in range(seq_decoder_input_reshape.size(1)):
                    decoder_hidden = torch.zeros((1, 256), device=gpu_device)
                    word = []
                    char_scores = []
                    detailed_char_scores = []
                    top_seqs = self.beam_search(
                        seq_decoder_input_reshape[:, batch_index: batch_index + 1, :],
                        decoder_hidden,
                        beam_size=6,
                        max_len=self.convert.batch_max_lenght,
                    )
                    top_seq = top_seqs[0]
                    for character in top_seq[1:]:
                        character_index = character[0]
                        if character_index == self.convert.dict['[s]']:
                            char_scores.append(character[1])
                            detailed_char_scores.append(character[2])
                            break
                        else:
                            if character_index == 0:
                                word.append('[GO]')
                                char_scores.append(0.0)
                            else:
                                word.append(self.convert.decode(character_index))
                                char_scores.append(character[1])
                                detailed_char_scores.append(character[2])
                    words.append("".join(word))
                    decoded_scores.append(char_scores)
                    detailed_decoded_scores.append(detailed_char_scores)
            else:
                for batch_index in range(seq_decoder_input_reshape.size(1)):
                    bos_onehot = np.zeros((1, 1), dtype=np.int32)
                    bos_onehot[:, 0] = self.convert.dict['[GO]']
                    decoder_input = torch.tensor(bos_onehot.tolist(), device=gpu_device)
                    decoder_hidden = torch.zeros((1, 256), device=gpu_device)
                    word = []
                    char_scores = []
                    for di in range(self.convert.batch_max_lenght):
                        decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                            decoder_input,
                            decoder_hidden,
                            seq_decoder_input_reshape[:, batch_index: batch_index + 1, :],
                        )
                        # decoder_attentions[di] = decoder_attention.data
                        maxk = max((1,))
                        topv, topi = decoder_output.topk(maxk, dim=1)
                        char_scores.append(topv.item())
                        if topi.item() == self.convert.dict['[s]']:
                            break
                        else:
                            if topi.item() == 0:
                                word.append('[GO]')
                            else:
                                word.append(self.convert.decode(topi.item()))

                        # real_length = di
                        decoder_input = topi.squeeze(1).detach()
                    words.append("".join(word))
                    decoded_scores.append(char_scores)
            return words, decoded_scores, detailed_decoded_scores

    def beam_search_step(self, encoder_context, top_seqs, k):
        all_seqs = []
        for seq in top_seqs:
            seq_score = reduce_mul([_score for _, _score, _, _ in seq])
            if seq[-1][0] == self.NUM_CHAR - 1:
                all_seqs.append((seq, seq_score, seq[-1][2], True))
                continue
            decoder_hidden = seq[-1][-1][0]
            onehot = np.zeros((1, 1), dtype=np.int32)
            onehot[:, 0] = seq[-1][0]
            decoder_input = torch.tensor(onehot.tolist(), device=gpu_device)
            decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                decoder_input, decoder_hidden, encoder_context
            )
            detailed_char_scores = decoder_output.cpu().numpy()
            scores, candidates = decoder_output.data[:, 1:].topk(k)
            for i in range(k):
                character_score = scores[:, i]
                character_index = candidates[:, i]
                score = seq_score * character_score.item()
                char_score = seq_score * detailed_char_scores
                rs_seq = seq + [
                    (
                        character_index.item() + 1,
                        character_score.item(),
                        char_score,
                        [decoder_hidden],
                    )
                ]
                done = character_index.item() + 1 == self.NUM_CHAR - 1
                all_seqs.append((rs_seq, score, char_score, done))
        all_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)
        topk_seqs = [seq for seq, _, _, _ in all_seqs[:k]]
        all_done = check_all_done(all_seqs[:k])
        return topk_seqs, all_done

    def beam_search(self, encoder_context, decoder_hidden, beam_size=6, max_len=32):
        char_score = np.zeros(self.NUM_CHAR)
        top_seqs = [[(self.convert.dict['[GO]'], 1.0, char_score, [decoder_hidden])]]
        # loop
        for _ in range(max_len):
            top_seqs, all_done = self.beam_search_step(
                encoder_context, top_seqs, beam_size
            )
            if all_done:
                break
        return top_seqs

    def rectification_pt(self, feat_map, ctps): # ctps : B * N  * 4
        batch_size, _, feat_height, feat_width = feat_map.size()
        grid = self.target_grid.repeat(batch_size, 1, 1, 1) # batch_size * 3 * h * w
        cropped_grid = torch.chunk(grid, self.num_of_slices, -1)
        cropped_grid = torch.stack(cropped_grid, dim=-1).permute(4, 0, 1, 2, 3).contiguous()
        B = batch_size * self.num_of_slices
        cropped_grid = cropped_grid.view(B, 3, feat_height, feat_width // self.num_of_slices)



        target_points = self.target_ctps.unsqueeze(0).repeat(batch_size, 1, 1) # batch_size * num_of_ctps * 4

        print(target_points.size()) #batch_size * control_point_num * 4
        print(target_points)

        target_points = torch.cat([target_points[:, 0:-1].view(batch_size, self.num_of_slices, 2, 2), target_points[:, 1:].view(batch_size, self.num_of_slices, 2, 2)], 2)\
            .transpose(0, 1).contiguous().view(-1, 4, 2)
        raw_points = torch.cat([ctps[:, 0:-1].view(batch_size, self.num_of_slices, 2, 2), ctps[:, 1:].view(batch_size, self.num_of_slices, 2, 2)], 2)\
            .transpose(0, 1).contiguous().view(-1, 4, 2)

        A = torch.zeros((B, 8, 8), device=gpu_device)
        A[:, 0:-1:2, 0] = raw_points[:, :, 0]
        A[:, 0:-1:2, 1] = raw_points[:, :, 1]
        A[:, 0:-1:2, 2] = torch.ones(B, 4)
        A[:, 1::2, 3] = raw_points[:, :, 0]
        A[:, 1::2, 4] = raw_points[:, :, 1]
        A[:, 1::2, 5] = torch.ones(B, 4)
        A[:, 0:-1:2, 6] = -torch.mul(raw_points[:, :, 0], target_points[:, :, 0])
        A[:, 0:-1:2, 7] = -torch.mul(raw_points[:, :, 1], target_points[:, :, 0])
        A[:, 1::2, 6] = -torch.mul(raw_points[:, :, 0], target_points[:, :, 1])
        A[:, 1::2, 7] = -torch.mul(raw_points[:, :, 1], target_points[:, :, 1])
        b = target_points.view(B, 8, 1)
        b_ = torch.matmul(torch.inverse(A), b).squeeze()
        H = torch.ones((B, 9), device=gpu_device)
        H[:, 0:-1] = b_
        H = H.view(B, 3, 3)
        rectified_cropped_grid = torch.matmul(H.inverse(), cropped_grid.view(B, 3, -1))
        rectified_cropped_grid_scale = torch.div(rectified_cropped_grid[:, 0:2, :], rectified_cropped_grid[:, -1, :].unsqueeze(1))
        rectified_cropped_grid_scale = rectified_cropped_grid_scale.view(B, 2, feat_height, feat_width // self.num_of_slices).permute(0, 2, 3, 1)
        rectified_cropped_grid_scale = torch.clamp(rectified_cropped_grid_scale, 0, 1) * 2. - 1
        rectified_grid = rectified_cropped_grid_scale.view(self.num_of_slices, batch_size, feat_height, feat_width // self.num_of_slices, 2)
        rectified_grid = rectified_grid.permute(1, 2, 0, 3, 4).contiguous().view(batch_size, feat_height, feat_width, 2)
        return F.grid_sample(feat_map, rectified_grid)


    def rectification_pt_2(self, feat_map, ctps): # ctps : B * N  * 4
        target_points_bia = ctps[1]
        ctps = ctps[0]
        #print(target_points_bia.size())
        #print(target_points_bia)
        print(torch.mean(torch.abs(target_points_bia),dim=1))
        target_points_bias = torch.zeros(target_points_bia.size(0),target_points_bia.size(1)+2, device=gpu_device)
        target_points_bias[:,1:-1] = target_points_bia
        target_points_zero = torch.zeros_like(target_points_bias, device=gpu_device)

        target_points_bias = torch.stack([target_points_bias,target_points_zero,target_points_bias,target_points_zero], dim=2)

        #print(target_points_bias.size())
        #print(target_points_bias)
        batch_size, _, feat_height, feat_width = feat_map.size()
        grid = self.target_grid.repeat(batch_size, 1, 1, 1) # batch_size * 3 * h * w
        cropped_grid = torch.chunk(grid, self.num_of_slices, -1)
        cropped_grid = torch.stack(cropped_grid, dim=-1).permute(4, 0, 1, 2, 3).contiguous()
        B = batch_size * self.num_of_slices
        cropped_grid = cropped_grid.view(B, 3, feat_height, feat_width // self.num_of_slices)



        target_points = self.target_ctps.unsqueeze(0).repeat(batch_size, 1, 1) # batch_size * num_of_ctps * 4


        target_points = target_points + target_points_bias
        #print(target_points.size()) #batch_size * control_point_num * 4
        #print(target_points)

        target_points = torch.cat([target_points[:, 0:-1].view(batch_size, self.num_of_slices, 2, 2), target_points[:, 1:].view(batch_size, self.num_of_slices, 2, 2)], 2)\
            .transpose(0, 1).contiguous().view(-1, 4, 2)
        raw_points = torch.cat([ctps[:, 0:-1].view(batch_size, self.num_of_slices, 2, 2), ctps[:, 1:].view(batch_size, self.num_of_slices, 2, 2)], 2)\
            .transpose(0, 1).contiguous().view(-1, 4, 2)

        A = torch.zeros((B, 8, 8), device=gpu_device)
        A[:, 0:-1:2, 0] = raw_points[:, :, 0]
        A[:, 0:-1:2, 1] = raw_points[:, :, 1]
        A[:, 0:-1:2, 2] = torch.ones(B, 4)
        A[:, 1::2, 3] = raw_points[:, :, 0]
        A[:, 1::2, 4] = raw_points[:, :, 1]
        A[:, 1::2, 5] = torch.ones(B, 4)
        A[:, 0:-1:2, 6] = -torch.mul(raw_points[:, :, 0], target_points[:, :, 0])
        A[:, 0:-1:2, 7] = -torch.mul(raw_points[:, :, 1], target_points[:, :, 0])
        A[:, 1::2, 6] = -torch.mul(raw_points[:, :, 0], target_points[:, :, 1])
        A[:, 1::2, 7] = -torch.mul(raw_points[:, :, 1], target_points[:, :, 1])
        b = target_points.view(B, 8, 1)
        b_ = torch.matmul(torch.inverse(A), b).squeeze()
        H = torch.ones((B, 9), device=gpu_device)
        H[:, 0:-1] = b_
        H = H.view(B, 3, 3)
        rectified_cropped_grid = torch.matmul(H.inverse(), cropped_grid.view(B, 3, -1))
        rectified_cropped_grid_scale = torch.div(rectified_cropped_grid[:, 0:2, :], rectified_cropped_grid[:, -1, :].unsqueeze(1))
        rectified_cropped_grid_scale = rectified_cropped_grid_scale.view(B, 2, feat_height, feat_width // self.num_of_slices).permute(0, 2, 3, 1)
        rectified_cropped_grid_scale = torch.clamp(rectified_cropped_grid_scale, 0, 1) * 2. - 1
        rectified_grid = rectified_cropped_grid_scale.view(self.num_of_slices, batch_size, feat_height, feat_width // self.num_of_slices, 2)
        rectified_grid = rectified_grid.permute(1, 2, 0, 3, 4).contiguous().view(batch_size, feat_height, feat_width, 2)
        return F.grid_sample(feat_map, rectified_grid)



    def rectification_tps(self, feat_map, ctps): # ctps : B * N  * 4
        batch_size, _, feat_height, feat_width = feat_map.size()
        ctps = torch.cat((ctps[:, :, 0:2], ctps[:, :, 2:]), 1) # B * 2N * 2
        Y = torch.cat((ctps, self.padding_matrix.expand(batch_size, 3, 2)), 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)

        grid = source_coordinate.view(-1, self.targetH, self.targetW, 2)
        grid = torch.clamp(grid, 0, 1) # the source_control_points may be out of [0, 1].
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(feat_map, grid, canvas=None)
        return output_maps


class Attn(nn.Module):
    def __init__(self, method, hidden_size, embed_size, onehot_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attn = nn.Linear(2 * self.hidden_size, hidden_size)
        # self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (B, hidden_size)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (H*W, B, hidden_size)
        :return
            attention energies in shape (B, H*W)
        """
        max_len = encoder_outputs.size(0)
        # this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # (B, H*W, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, H*W, hidden_size)
        attn_energies = self.score(H, encoder_outputs)  # compute attention score (B, H*W)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax (B, 1, H*W)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # (B, H*W, 2*hidden_size+H+W)->(B, H*W, hidden_size)
        energy = energy.transpose(2, 1)  # (B, hidden_size, H*W)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # (B, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (B, 1, H*W)
        return energy.squeeze(1)  # (B, H*W)


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0., bidirectional=False, onehot_size = (8, 32)):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.embedding.weight.data = torch.eye(embed_size)
        # self.dropout = nn.Dropout(dropout_p)
        self.word_linear = nn.Linear(embed_size, hidden_size)
        self.attn = Attn("concat", hidden_size, embed_size, onehot_size[0] + onehot_size[1])
        self.rnn = nn.GRUCell(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        """
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B, hidden_size)
        :param encoder_outputs:
            encoder outputs in shape (H*W, B, C)
        :return
            decoder output
        """
        # Get the embedding of the current input word (last output word)
        word_embedded_onehot = self.embedding(word_input).view(1, word_input.size(0), -1)  # (1,B,embed_size)
        word_embedded = self.word_linear(word_embedded_onehot)  # (1, B, hidden_size)
        attn_weights = self.attn(last_hidden, encoder_outputs)  # (B, 1, H*W)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B, 1, H*W) * (B, H*W, C) = (B,1,C)
        context = context.transpose(0, 1)  # (1,B,C)
        # Combine embedded input word and attended context, run through RNN
        # 2 * hidden_size + W + H: 256 + 256 + 32 + 8 = 552
        rnn_input = torch.cat((word_embedded, context), dim=2)
        last_hidden = last_hidden.view(last_hidden.size(0), -1)
        rnn_input = rnn_input.view(word_input.size(0), -1)
        hidden = self.rnn(rnn_input, last_hidden)
        if not self.training:
            output = F.softmax(self.out(hidden), dim=1)
        else:
            output = F.log_softmax(self.out(hidden), dim=1)
        # Return final output, hidden state
        return output, hidden, attn_weights


# def make_roi_seq_predictor(cfg, dim_in):
#     return SequencePredictor(cfg, dim_in)