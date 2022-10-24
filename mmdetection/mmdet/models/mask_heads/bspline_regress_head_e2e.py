import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from functools import reduce
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import regress_point_target, force_fp32, auto_fp16
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F


class Interval(object):
    """Numeric interval [a,b].

    Attributes:
        a -- min value
        b -- max value
    """

    def __init__(self, a, b):
        super(Interval, self).__init__()

        assert (a <= b)

        self.a = a
        self.b = b

    def linspace(self, n):
        return np.linspace(self.a, self.b, n)

    def __contains__(self, x):
        return self.a <= x <= self.b

    def __eq__(self, another):
        return self.a == another.a and self.b == another.b

    def __lt__(self, x):
        return x < self.a

    def __gt__(self, x):
        return x > self.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __str__(self):
        return '[%f,%f]' % (self.a, self.b)


class BSpline(object):
    """Clamped b-spline is connected to the first and last points.

    Attributes:
        points -- interpolation points
        p -- spline degree
        C -- evaluation functions
        N -- basis functions
        U -- knots

    TODO: return first or last function if value is not in range
    """

    def __init__(self, num_points, p, *args,**kwargs):
        super(BSpline, self).__init__(*args,**kwargs)

        assert (2 <= num_points)
        assert (0 <= p and p <= num_points - 1)

        # self.points = np.asarray(points)
        self.num_points = num_points
        self.p = p

        self.U = self._uniform_knots(self.n, self.p)
        self.N = self._basis(self.U, self.p)

        self.C = {}
        # for i,poly in self.N.items():
        #     self.C[i] = [reduce(np.polyadd, (np.polymul(p,c) for p,c in zip(poly,coord))) for coord in np.transpose(self.points)]

    def _uniform_knots(self, n, p):
        U = [(i + 1) / (n - p) for i in range(n - p - 1)]
        return np.concatenate([np.zeros(p + 1), U, np.full(p + 1, 1)])

    def _Nij(self, N, i, j, U):
        if not np.any(N[i]) and not np.any(N[i + 1]):
            return np.array([0])
        elif not np.any(N[i]):
            return np.polymul(N[i + 1], [-1, U[i + j + 1]]) / (U[i + j + 1] - U[i + 1])
        elif not np.any(N[i + 1]):
            return np.polymul(N[i], [1, -U[i]]) / (U[i + j] - U[i])
        return np.polyadd(np.polymul(N[i], [1, -U[i]]) / (U[i + j] - U[i]), np.polymul(N[i + 1], [-1, U[i + j + 1]]) / (U[i + j + 1] - U[i + 1]))

    def _basis(self, U, p):
        res = {}
        for i in (i for i in range(len(U) - 1) if U[i] != U[i + 1]):
            N = [[0] for i in range(len(U) - 1)]
            N[i][0] = 1
            for j in range(1, p + 1):
                N[:] = [self._Nij(N, i, j, U) for i in range(len(N) - 1)]
            res[Interval(U[i], U[i + 1])] = np.asarray(N)
        return res

    def _find_interval(self, u):
        for i, c in self.C.items():
            if u in i:
                return c
        raise Exception

    def eval2d(self, u):
        f = self._find_interval(u)
        return tuple(np.polyval(c, u) for c in f[:2])

    def eval(self, u):
        f = self._find_interval(u)
        return tuple(np.polyval(c, u) for c in f)

    def evalx(self, u):
        f = self._find_interval(u)
        return np.polyval(f[0], u)

    def evaly(self, u):
        f = self._find_interval(u)
        return np.polyval(f[1], u)

    def evalz(self, u):
        # if (len(self.points[0]) < 3):
        #    return 0
        f = self._find_interval(u)
        return np.polyval(f[2], u)

    #
    # """x-values of width points."""
    # @property
    # def X(self):
    #     return [p[0] for p in self.points]
    #
    # """y-values of width points."""
    # @property
    # def Y(self):
    #     return [p[1] for p in self.points]
    #
    # """z-values of width points."""
    # @property
    # def Z(self):
    #     if (len(self.points[0]) > 2):
    #         return [p[2] for p in self.points]
    #     return np.zeros(len(self.points))

    """B-Spline parameter domain."""

    @property
    def domain(self):
        U = self.knots
        return Interval(np.min(U), np.max(U))

    """Number of points."""

    @property
    def n(self):
        return self.num_points

    """Number of knots."""

    @property
    def m(self):
        return self.n + self.p + 1

    """Knots."""

    @property
    def knots(self):
        return np.unique(self.U)


@HEADS.register_module
class FCNBSplineRegressE2EHead(nn.Module, BSpline):

    def __init__(self,
                 num_convs=3,
                 roi_feat_size=16,
                 in_channels=256,
                 conv_kernel_size=3,
                 use_anchor_t=True,
                 b_spline_degree=3,
                 num_control_point=5,
                 num_width_line=5,
                 num_fit_point=16,
                 num_classes=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_avg_pool=True,
                 use_poly_loss=False,
                 max_batch_size=2000,
                 loss_regress_point=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_cls=dict(
                     type='DirCrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_poly=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=10)
                 ):
        nn.Module.__init__(self)
        BSpline.__init__(self, num_points=num_control_point, p=b_spline_degree)
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = [256, 256, 512]
        self.fc_out_channels = [1024, 38]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.b_spline_degree = b_spline_degree
        self.num_control_point = num_control_point
        self.num_width_line = num_width_line
        self.num_fit_point = num_fit_point
        self.loss_regress_point = build_loss(loss_regress_point)
        self.loss_direction = build_loss(loss_cls)
        self.loss_border = build_loss(loss_poly)
        self.with_avg_pool = with_avg_pool
        self.num_classes = num_classes
        self.max_batch_size = max_batch_size
        self.use_poly_loss = use_poly_loss
        if self.with_avg_pool:
            self.avg_pool_cls = nn.AvgPool2d(roi_feat_size)
        self.fc_cls = nn.Linear(in_channels, 2)

        for i, poly in self.N.items():
            N = []
            for p in poly:
                if len(p) < self.b_spline_degree + 1:
                    p = np.concatenate((np.zeros((self.b_spline_degree + 1 - len(p))), p), axis=0)
                N.append(p)
            N = torch.from_numpy(np.array(N))
            N = N.expand(self.max_batch_size, N.size(0), N.size(1)).float().transpose(1, 2)
            self.N[i] = N

        anchor_t = torch.from_numpy(np.linspace(0., 1., self.num_fit_point)).float()
        self.anchor_t = anchor_t.expand(self.max_batch_size, anchor_t.size(0))

        anchor_border_t = torch.from_numpy(np.linspace(0., 1., self.num_width_line)).float()
        self.anchor_border_t = anchor_border_t.expand(self.max_batch_size, anchor_border_t.size(0))

        self.use_anchor_t = use_anchor_t

        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2))

        self.num_param = self.num_control_point * 2 + self.num_fit_point + self.num_width_line * 2 * 4
        self.fc_out_channels[-1] = self.num_param


        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels[i - 1])
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels[i],
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            if i != 0:
                self.convs.append(
                    self.avg_pool
                )

        self.branch_fcs = nn.ModuleList()
        num_branch_fcs = 2
        last_layer_dim = 8192
        if num_branch_fcs > 0:
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels[i - 1])
                self.branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels[i]))
                if i!= num_branch_fcs-1:
                    self.branch_fcs.append(nn.LeakyReLU())

        self.fc_out_channels_rec = 7
        self.branch_rec = nn.ModuleList()
        last_layer_dim_rec = 8192
        if num_branch_fcs > 0:
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim_rec if i == 0 else self.fc_out_channels_rec)
                self.branch_rec.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels_rec))
        self.debug_imgs = None

    def init_weights(self):
        # super(FCNLineRegressHead, self).init_weights()
        for module_list in [self.convs]:
            for m in module_list.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        for module_list in [self.branch_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x_cls = self.avg_pool_cls(x)
        else:
            x_cls = x.clone()
        x_cls = x_cls.view(x_cls.size(0), -1)
        x_cls = self.fc_cls(x_cls)
        for conv in self.convs:
            x = conv(x)
        x_pre = x.view(x.size(0), -1)
        for fc in self.branch_fcs:
            x = fc(x_pre)
        for fc in self.branch_rec:
            x_1 = fc(x_pre)
        x = torch.cat([x, x_1], dim=-1).view(x.size(0), -1)
        return x, x_cls

    def get_target(self, sampling_results, gt_points, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        regress_point_targets = regress_point_target(pos_proposals, pos_assigned_gt_inds,
                                                     gt_points, rcnn_train_cfg)
        return regress_point_targets

    def change_format(self, x):
        change_val = torch.stack([x ** i for i in reversed(range(0, self.b_spline_degree + 1))], dim=-1)
        return change_val

    @force_fp32(apply_to=('para_lines_pred', 'centern_line_targets', 'width_line_targets', 'width_line_angle_targets', 'width_line_x_target', 'width_line_y_target',
                          'width_line_length_targets', 'poly_pts_targets'))
    def loss(self, para_lines_pred, direction_pred, centern_line_targets, width_line_targets, width_line_angle_targets,
                   width_line_x_target, width_line_y_target, width_line_length_targets, direction_targets, poly_pts_targets, pos_rois, labels):
        # torch.set_printoptions(threshold=np.nan)


        bbox = pos_rois[:, 1:]
        w = bbox[:, 2] - bbox[:, 0] + 1
        h = bbox[:, 3] - bbox[:, 1] + 1
        bbox_centern_x = (bbox[:, 0] + bbox[:, 2]) / 2
        bbox_centern_y = (bbox[:, 1] + bbox[:, 3]) / 2

        centern_line_targets[:, :, 0] -= torch.unsqueeze(bbox_centern_x, dim=-1).expand(bbox_centern_x.size(0), centern_line_targets.size(1))
        centern_line_targets[:, :, 1] -= torch.unsqueeze(bbox_centern_y, dim=-1).expand(bbox_centern_y.size(0), centern_line_targets.size(1))
        centern_line_targets[:, :, 0] /= torch.unsqueeze(w, dim=-1).expand(w.size(0), centern_line_targets.size(1))
        centern_line_targets[:, :, 1] /= torch.unsqueeze(h, dim=-1).expand(h.size(0), centern_line_targets.size(1))
        # print_width_line_targets = width_line_targets.clone()
        width_line_targets[:, :, :, 0] -= torch.unsqueeze(bbox_centern_x, dim=-1).unsqueeze(dim=-1).expand(bbox_centern_x.size(0), width_line_targets.size(1), width_line_targets.size(2))
        width_line_targets[:, :, :, 1] -= torch.unsqueeze(bbox_centern_y, dim=-1).unsqueeze(dim=-1).expand(bbox_centern_y.size(0), width_line_targets.size(1), width_line_targets.size(2))
        width_line_targets[:, :, :, 0] /= torch.unsqueeze(w, dim=-1).unsqueeze(dim=-1).expand(w.size(0), width_line_targets.size(1), width_line_targets.size(2))
        width_line_targets[:, :, :, 1] /= torch.unsqueeze(h, dim=-1).unsqueeze(dim=-1).expand(h.size(0), width_line_targets.size(1), width_line_targets.size(2))

        delta_x = width_line_targets[:, :, 0, 0] - width_line_targets[:, :, 2, 0]
        delta_y = width_line_targets[:, :, 0, 1] - width_line_targets[:, :, 2, 1]

        direction_targets = torch.squeeze(direction_targets, dim=1).long()
        delta = torch.stack((delta_x, delta_y), dim=-1)
        delta_ind = direction_targets.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(direction_targets.size(0), delta.size(1), 1)
        delta_x_coll = torch.gather(delta, dim=2, index=delta_ind).squeeze(dim=2)
        delta_y_coll = torch.gather(delta, dim=2, index=1-delta_ind).squeeze(dim=2)

        # delta_x_coll = torch.stack([delta_x[i] if direction_targets[i][0] == 0 else delta_y[i] for i in range(para_lines_pred.size(0))],dim=0)
        # delta_y_coll = torch.stack([delta_y[i] if direction_targets[i][0] == 0 else delta_x[i] for i in range(para_lines_pred.size(0))],dim=0)

        width_line_angle_targets = torch.atan(torch.div(delta_x_coll, delta_y_coll))

        width_line_length_targets = (width_line_targets[:, :, 0, :] - width_line_targets[:, :, 2, :]).pow(2)
        width_line_length_targets = torch.sum(width_line_length_targets, dim=-1).pow(0.5)
        poly_pts_targets = torch.cat((poly_pts_targets[:, :, 0, :], poly_pts_targets[:, :, 1, :].flip((1,))), dim=1)
        poly_pts_targets[:, :, 0] -= bbox_centern_x.unsqueeze(dim=-1).expand(bbox_centern_x.size(0), poly_pts_targets.size(1))
        poly_pts_targets[:, :, 1] -= bbox_centern_y.unsqueeze(dim=-1).expand(bbox_centern_y.size(0), poly_pts_targets.size(1))
        poly_pts_targets[:, :, 0] /= w.unsqueeze(dim=-1).expand(w.size(0), poly_pts_targets.size(1))
        poly_pts_targets[:, :, 1] /= h.unsqueeze(dim=-1).expand(h.size(0), poly_pts_targets.size(1))

        control_line_d_pred = para_lines_pred[:, self.num_param * 2:].view(para_lines_pred.size(0), 7)
        control_line_d_pred = F.tanh(control_line_d_pred) * 0.0625
        para_lines_pred = para_lines_pred[:, :self.num_param * 2].view(para_lines_pred.size(0),self.num_param)

        pred_control_point = para_lines_pred[:, 0:self.num_control_point * 2].view(para_lines_pred.size(0), self.num_control_point, 2)

        pred_fit_para = para_lines_pred[:, self.num_control_point * 2:self.num_control_point * 2 + self.num_fit_point]
        pred_fit_para = torch.sigmoid(pred_fit_para)

        para_lines_angle_l = para_lines_pred[:, self.num_control_point * 2 + self.num_fit_point:].view(para_lines_pred.size(0), -1, self.num_width_line * 4)
        para_lines_angle_l_ind = direction_targets.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(direction_targets.size(0), 1 ,para_lines_angle_l.size(2))
        para_lines_angle_l = torch.gather(para_lines_angle_l, dim=1, index=para_lines_angle_l_ind).squeeze(dim=1)
        # para_lines_angle_l = torch.stack([para_lines_angle_l[i, 0, :] if direction_targets[i][0] == 0. else para_lines_angle_l[i, 1, :] for i in range(para_lines_pred.size(0))], dim=0)
        para_width_line_t = torch.stack([para_lines_angle_l[:, 4 * i] for i in range(self.num_width_line)], dim=1)
        para_width_line_t = torch.sigmoid(para_width_line_t)
        para_width_line_angel_pred = torch.stack([para_lines_angle_l[:, 1 + 4 * i] for i in range(self.num_width_line)], dim=1)
        para_width_line_length_pred = torch.cat([para_lines_angle_l[:, 2 + 4 * i:4 + 4 * i] for i in range(self.num_width_line)], dim=1)

        if not self.use_anchor_t:
            pred_fit_para_first_last=torch.cat((pred_fit_para[:,0:1],pred_fit_para[:,-1:]),dim=-1)
            pred_fit_para=torch.cat((para_lines_pred.new_zeros((para_lines_pred.size(0),1),requires_grad=False),pred_fit_para[:,1:-1],para_lines_pred.new_ones((para_lines_pred.size(0),1),requires_grad=False)),dim=-1)
            pred_border_fit_para_first_last=torch.cat((para_width_line_t[:,0:1],para_width_line_t[:,-1:]),dim=-1)
            para_width_line_t=torch.cat((para_lines_pred.new_zeros((para_lines_pred.size(0),1),requires_grad=False),para_width_line_t[:,1:-1],para_lines_pred.new_ones((para_lines_pred.size(0),1),requires_grad=False)),dim=-1)

        pred_fit_para = torch.cat((pred_fit_para, para_width_line_t),dim=1)
        if self.use_anchor_t:
            anchor_t = self.anchor_t[:para_lines_pred.size(0), :].to(para_lines_pred.device)
            anchor_border_t=self.anchor_border_t[:para_lines_pred.size(0),:].to(para_lines_pred.device)
            anchor_t=torch.cat((anchor_t, anchor_border_t),dim=1)
            fit_matrix = self.change_format(anchor_t)
        else:
            fit_matrix = self.change_format(pred_fit_para)

        N_matrix = []
        for i, poly in self.N.items():
            poly = poly[:para_lines_pred.size(0), :, :].to(para_lines_pred.device)
            N_matrix.append(torch.bmm(fit_matrix, poly))
            # print('torch.bmm(fit_matrix, poly) {}' .format(torch.bmm(fit_matrix, poly).size()))
        M = torch.stack(N_matrix, dim=0).view(len(self.knots) - 1, -1, self.num_control_point)
        # print(M.size())
        # print('pred_fit_para {}'.format(pred_fit_para))
        flat_fit_para = pred_fit_para

        for i in range(len(self.knots) - 1):
            flat_fit_para = torch.where((flat_fit_para >= self.knots[i]) & (flat_fit_para <= self.knots[i + 1]), torch.full_like(flat_fit_para, i), flat_fit_para)
        # print('flat_fit_para {}'.format(flat_fit_para))
        flat_fit_para = flat_fit_para.view(-1).long()
        # M = torch.stack([M[flat_fit_para[i], i, :] for i in range(flat_fit_para.size(0))]).view(para_lines_pred.size(0), self.num_fit_point + self.num_width_line, self.num_control_point)
        M_ind = flat_fit_para.unsqueeze(dim=0).unsqueeze(dim=-1).expand(1, flat_fit_para.size(0), M.size(2))
        M = torch.gather(M, dim=0, index=M_ind).squeeze(dim=0).view(para_lines_pred.size(0), self.num_fit_point + self.num_width_line, self.num_control_point)
        pred_fit_point = torch.bmm(M, pred_control_point)
        width_fit_point=pred_fit_point[:,self.num_fit_point:,:]
        pred_fit_point = pred_fit_point[:,:self.num_fit_point,:]

        width_line_length_targets = width_line_length_targets.unsqueeze(dim=-1).repeat(1, 1, 2).view(para_lines_pred.size(0), -1).div(2)
        if self.use_anchor_t:
            pred = torch.cat((pred_fit_point.view(para_lines_pred.size(0), -1), pred_fit_para), dim=1)
            targets = torch.cat((centern_line_targets.view(para_lines_pred.size(0), -1), anchor_t), dim=1)
        else:
            pred = torch.cat((pred_fit_point.view(para_lines_pred.size(0), -1), pred_fit_para_first_last, pred_border_fit_para_first_last), dim=1)
            targets = torch.cat((centern_line_targets.view(para_lines_pred.size(0), -1),
                                 torch.cat((para_lines_pred.new_zeros((para_lines_pred.size(0),1),requires_grad=False),para_lines_pred.new_ones((para_lines_pred.size(0),1),requires_grad=False)),dim=-1),
                                 torch.cat((para_lines_pred.new_zeros((para_lines_pred.size(0),1),requires_grad=False),para_lines_pred.new_ones((para_lines_pred.size(0),1),requires_grad=False)),dim=-1)), dim=1)
        loss = dict()
        weight = torch.ones_like(targets, requires_grad=False)
        for i in range(direction_targets.size(0)):
            if direction_targets[i] == 1:
                weight[i, self.num_fit_point * 2:] *= 5
        loss_regress_points = self.loss_regress_point(pred, targets, weight)
        loss['loss_center'] = loss_regress_points

        width_line_preds = torch.cat((para_width_line_angel_pred, para_width_line_length_pred), dim=-1)
        width_line_targets = torch.cat((width_line_angle_targets, width_line_length_targets), dim=-1)
        # print('width_line_targets {},--{},--{},--{},--{},--{}'.format(width_line_targets, delta_x, delta_y, direction_targets, print_width_line_targets, pos_rois))
        weight = torch.ones_like(width_line_preds, requires_grad=False)
        loss_width = self.loss_regress_point(width_line_preds, width_line_targets, weight)
        loss['loss_width'] = loss_width


        weight_cls = direction_pred.new_ones(2, requires_grad=False)
        weight_cls[1] = 5
        # labels = torch.squeeze(direction_targets, dim=-1).long()
        loss_direction = self.loss_direction(direction_pred, direction_targets, weight_cls)
        loss['loss_direction'] = loss_direction

        # if not self.use_poly_loss:
        #     return loss, pred_control_point.clone().detach()

        p = width_fit_point
        x = p[:, :, 0]
        y = p[:, :, 1]
        length_1 = para_width_line_length_pred[:, 0::2]
        length_2 = para_width_line_length_pred[:, 1::2]

        cos = torch.cos(para_width_line_angel_pred)
        sin = torch.sin(para_width_line_angel_pred)
        l1_sin = length_1 * sin
        l1_cos = length_1 * cos
        l2_sin = length_2 * sin
        l2_cos = length_2 * cos
        x_1_dir0 = x + l1_sin
        y_1_dir0 = y + l1_cos
        x_2_dir0 = x - l2_sin
        y_2_dir0 = y - l2_cos
        x_1_dir1 = x + l1_cos
        y_1_dir1 = y + l1_sin
        x_2_dir1 = x - l2_cos
        y_2_dir1 = y - l2_sin
        x_y_ind = direction_targets.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(direction_targets.size(0), x_1_dir0.size(1), 1)
        x_1 = torch.gather(torch.stack((x_1_dir0, x_1_dir1), dim=-1), dim=2, index=x_y_ind)
        y_1 = torch.gather(torch.stack((y_1_dir0, y_1_dir1), dim=-1), dim=2, index=x_y_ind)
        x_2 = torch.gather(torch.stack((x_2_dir0, x_2_dir1), dim=-1), dim=2, index=x_y_ind)
        y_2 = torch.gather(torch.stack((y_2_dir0, y_2_dir1), dim=-1), dim=2, index=x_y_ind)
        # x_1 = torch.stack([x[i, :] + length_1[i, :] * torch.sin(para_width_line_angel_pred[i, :]) if direction_targets[i][0] == 0 else x[i, :] + length_1[i, :] * torch.cos(para_width_line_angel_pred[i, :]) for i in
        #                    range(para_lines_pred.size(0))], dim=0)
        # y_1 = torch.stack([y[i, :] + length_1[i, :] * torch.cos(para_width_line_angel_pred[i, :]) if direction_targets[i][0] == 0 else y[i, :] + length_1[i, :] * torch.sin(para_width_line_angel_pred[i, :]) for i in
        #                    range(para_lines_pred.size(0))], dim=0)
        # x_2 = torch.stack([x[i, :] - length_2[i, :] * torch.sin(para_width_line_angel_pred[i, :]) if direction_targets[i][0] == 0 else x[i, :] - length_2[i, :] * torch.cos(para_width_line_angel_pred[i, :]) for i in
        #                    range(para_lines_pred.size(0))], dim=0)
        # y_2 = torch.stack([y[i, :] - length_2[i, :] * torch.cos(para_width_line_angel_pred[i, :]) if direction_targets[i][0] == 0 else y[i, :] - length_2[i, :] * torch.sin(para_width_line_angel_pred[i, :]) for i in
        #                    range(para_lines_pred.size(0))], dim=0)
        points = torch.cat((x_1, y_1, x.unsqueeze(dim=-1), y.unsqueeze(dim=-1), x_2, y_2), dim=-1).view(x.size(0), x.size(1), 3, 2)
        direction = []
        for i in range(points.size(1)):
            if i != points.size(1) - 1:
                mid = points[:, i + 1, 1, :] - points[:, i, 1, :]
                p1 = points[:, i, 0, :] - points[:, i, 1, :]
                p2 = points[:, i, 2, :] - points[:, i, 1, :]
                direction.append((p1[:, 0] - mid[:, 0]) * (p2[:, 1] - mid[:, 1]) - (p2[:, 0] - mid[:, 0]) * (p1[:, 1] - mid[:, 1]))
            else:
                mid = points[:, i, 1, :] - points[:, i - 1, 1, :]
                p1 = points[:, i, 0, :] - points[:, i, 1, :]
                p2 = points[:, i, 2, :] - points[:, i, 1, :]
                direction.append((p1[:, 0] - mid[:, 0]) * (p2[:, 1] - mid[:, 1]) - (p2[:, 0] - mid[:, 0]) * (p1[:, 1] - mid[:, 1]))
        direction = torch.stack(direction, dim=-1)

        for_index = torch.where(direction > 0, torch.full_like(direction, 0), torch.full_like(direction, 2))
        for_index = for_index.long()
        for_index = torch.unsqueeze(for_index, dim=-1)
        for_index = torch.unsqueeze(for_index, dim=-1).expand(for_index.size(0), for_index.size(1), 1, points.size(3))

        for_point = torch.gather(points, dim=2, index=for_index)
        for_point = torch.squeeze(for_point, dim=2)

        back_index = torch.where(direction > 0, torch.full_like(direction, 2), torch.full_like(direction, 0))
        back_index = back_index.long()
        back_index = torch.unsqueeze(back_index, dim=-1)
        back_index = torch.unsqueeze(back_index, dim=-1)
        back_index = back_index.expand(back_index.size(0), back_index.size(1), 1, points.size(3))
        back_point = torch.gather(points, dim=2, index=back_index)
        back_point = torch.squeeze(back_point, dim=2)
        back_point = torch.flip(back_point, (1,))
        result = torch.cat((for_point, back_point), dim=1)
        result = result.view(para_lines_pred.size(0), -1)

        return_pts_gt = poly_pts_targets.clone().detach()
        return_pts_gt = torch.chunk(return_pts_gt, chunks=2, dim=1)
        return_pts_gt = torch.cat((return_pts_gt[0].flip((1,)), return_pts_gt[1]), dim=-1)
        if not self.use_poly_loss:
            return loss, (return_pts_gt + 0.5, control_line_d_pred)

        poly_pts_targets = poly_pts_targets.view(para_lines_pred.size(0), -1)
        border_preds = result
        border_targets = poly_pts_targets
        # border_preds = torch.cat((result, para_width_line_angel_pred, para_width_line_length_pred), dim=-1)
        # border_targets = torch.cat((poly_pts_targets, width_line_angle_targets, width_line_length_targets), dim=-1)
        weight = torch.ones_like(border_targets, requires_grad=False)
        loss_border = self.loss_border(border_preds, border_targets, weight)
        loss['loss_border'] = loss_border
        return loss, pred_control_point.clone().detach()
        # return loss, return_pts_gt + 0.5

    def get_seg_masks(self, para_regress_point, direction_pred, det_bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        ind = torch.argmax(direction_pred, dim=1)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        control_line_d_pred = para_regress_point[:, self.num_param: ].view(para_regress_point.size(0), 7)
        para_regress_point = para_regress_point[:, :self.num_param].view(para_regress_point.size(0), self.num_param)

        control_line_d_pred = F.tanh(control_line_d_pred) * 0.0625

        pred_control_point = para_regress_point[:, 0:self.num_control_point * 2].view(para_regress_point.size(0), self.num_control_point, 2)
        pred_fit_para = para_regress_point[:, self.num_control_point * 2:self.num_control_point * 2 + self.num_fit_point]
        pred_fit_para = torch.sigmoid(pred_fit_para)

        para_lines_angle_l = para_regress_point[:, self.num_control_point * 2 + self.num_fit_point:].view(para_regress_point.size(0), -1, self.num_width_line * 4)
        para_lines_angle_l = torch.stack([para_lines_angle_l[i, ind[i], :] for i in range(para_regress_point.size(0))], dim=0)

        para_width_line_t = torch.stack([para_lines_angle_l[:, 4 * i] for i in range(self.num_width_line)], dim=1)
        pred_fit_para = torch.sigmoid(para_width_line_t)
        para_width_line_angel_pred = torch.stack([para_lines_angle_l[:, 1+4 * i] for i in range(self.num_width_line)], dim=1)
        para_width_line_length_pred = torch.cat([para_lines_angle_l[:, 2 + 4 * i:4 + 4 * i] for i in range(self.num_width_line)], dim=1)

        if not self.use_anchor_t:
            pred_fit_para = torch.cat((para_regress_point.new_zeros((para_regress_point.size(0), 1), requires_grad=False), pred_fit_para[:, 1:-1],
                                       para_regress_point.new_ones((para_regress_point.size(0), 1), requires_grad=False)), dim=-1)

        fit_matrix = self.change_format(pred_fit_para)
        N_matrix = []
        for i, poly in self.N.items():
            poly = poly[:para_regress_point.size(0), :, :].to(para_regress_point.device)
            N_matrix.append(torch.bmm(fit_matrix, poly))

        M = torch.stack(N_matrix, dim=0).view(len(self.knots) - 1, -1, self.num_control_point)
        flat_fit_para = pred_fit_para
        for i in range(len(self.knots) - 1):
            flat_fit_para = torch.where((flat_fit_para >= self.knots[i]) & (flat_fit_para <= self.knots[i + 1]), torch.full_like(flat_fit_para, i), flat_fit_para)
        flat_fit_para = flat_fit_para.view(-1).long()
        M = torch.stack([M[flat_fit_para[i], i, :] for i in range(flat_fit_para.size(0))]).view(para_regress_point.size(0), self.num_width_line, self.num_control_point)
        pred_fit_point = torch.bmm(M, pred_control_point)
        x = pred_fit_point[:, :, 0]
        y = pred_fit_point[:, :, 1]

        length_1 = para_width_line_length_pred[:, 0::2]
        length_2 = para_width_line_length_pred[:, 1::2]
        x_1 = torch.stack([x[i, :] + length_1[i, :] * torch.sin(para_width_line_angel_pred[i, :]) if ind[i] == 0 else x[i, :] + length_1[i, :] * torch.cos(para_width_line_angel_pred[i, :]) for i in range(para_regress_point.size(0))], dim=0)
        y_1 = torch.stack([y[i, :] + length_1[i, :] * torch.cos(para_width_line_angel_pred[i, :]) if ind[i] == 0 else y[i, :] + length_1[i, :] * torch.sin(para_width_line_angel_pred[i, :]) for i in range(para_regress_point.size(0))], dim=0)
        x_2 = torch.stack([x[i, :] - length_2[i, :] * torch.sin(para_width_line_angel_pred[i, :]) if ind[i] == 0 else x[i, :] - length_2[i, :] * torch.cos(para_width_line_angel_pred[i, :]) for i in range(para_regress_point.size(0))], dim=0)
        y_2 = torch.stack([y[i, :] - length_2[i, :] * torch.cos(para_width_line_angel_pred[i, :]) if ind[i] == 0 else y[i, :] - length_2[i, :] * torch.sin(para_width_line_angel_pred[i, :]) for i in range(para_regress_point.size(0))], dim=0)

        points = torch.stack([x_1, y_1, x, y, x_2, y_2], dim=-1).view(x.size(0), x.size(1), 3, 2)

        direction = []
        for i in range(points.size(1)):
            if i != points.size(1) - 1:
                mid = points[:, i + 1, 1, :] - points[:, i, 1, :]
                p1 = points[:, i, 0, :] - points[:, i, 1, :]
                p2 = points[:, i, 2, :] - points[:, i, 1, :]
                direction.append((p1[:, 0] - mid[:, 0]) * (p2[:, 1] - mid[:, 1]) - (p2[:, 0] - mid[:, 0]) * (p1[:, 1] - mid[:, 1]))
            else:
                mid = points[:, i, 1, :] - points[:, i - 1, 1, :]
                p1 = points[:, i, 0, :] - points[:, i, 1, :]
                p2 = points[:, i, 2, :] - points[:, i, 1, :]
                direction.append((p1[:, 0] - mid[:, 0]) * (p2[:, 1] - mid[:, 1]) - (p2[:, 0] - mid[:, 0]) * (p1[:, 1] - mid[:, 1]))
        direction = torch.stack(direction, dim=-1)

        for_index = torch.where(direction > 0, torch.full_like(direction, 0), torch.full_like(direction, 2))
        for_index = for_index.long()
        for_index = torch.unsqueeze(for_index, dim=-1)
        for_index = torch.unsqueeze(for_index, dim=-1).expand(for_index.size(0), for_index.size(1), 1, points.size(3))

        for_point = torch.gather(points, dim=2, index=for_index)
        for_point = torch.squeeze(for_point, dim=2)

        back_index = torch.where(direction > 0, torch.full_like(direction, 2), torch.full_like(direction, 0))
        back_index = back_index.long()
        back_index = torch.unsqueeze(back_index, dim=-1)
        back_index = torch.unsqueeze(back_index, dim=-1)
        back_index = back_index.expand(back_index.size(0), back_index.size(1), 1, points.size(3))
        back_point = torch.gather(points, dim=2, index=back_index)
        back_point = torch.squeeze(back_point, dim=2)
        back_point = torch.flip(back_point, (1,))

        result = torch.cat((for_point, back_point), dim=1)
        return_pts = torch.cat((for_point.flip(dims=(1,)), back_point), dim=-1)

        bbox = det_bboxes / scale_factor
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = torch.unsqueeze(w, dim=-1).expand(w.size(0), result.size(1))

        h = bbox[:, 3] - bbox[:, 1] + 1
        h = torch.unsqueeze(h, dim=-1).expand(h.size(0), result.size(1))

        bbox_centern_x = (bbox[:, 0] + bbox[:, 2]) / 2
        bbox_centern_x = torch.unsqueeze(bbox_centern_x, dim=-1).expand(bbox_centern_x.size(0), result.size(1))

        bbox_centern_y = (bbox[:, 1] + bbox[:, 3]) / 2
        bbox_centern_y = torch.unsqueeze(bbox_centern_y, dim=-1).expand(bbox_centern_y.size(0), result.size(1))

        result[:, :, 0] *= w
        result[:, :, 1] *= h
        result[:, :, 0] += bbox_centern_x
        result[:, :, 1] += bbox_centern_y

        result = result.cpu().numpy().astype(np.int)
        # return result, return_pts + 0.5
        return result, (pred_control_point, control_line_d_pred)
