from .two_stage import TwoStageDetector
from ..registry import DETECTORS
import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler, bbox_mapping, merge_aug_bboxes, \
    merge_aug_masks, multiclass_nms


@DETECTORS.register_module
class TextRegressE2ECascadeRCNN(BaseDetector, RPNTestMixin):

    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 reg_roi_scale_factor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 recognition_head=None,
                 recognition_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(TextRegressE2ECascadeRCNN, self).__init__()

        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)
        self.reg_roi_scale_factor = reg_roi_scale_factor
        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        if recognition_head is not None:
            if recognition_roi_extractor is not None:
                self.recognition_roi_extractor = builder.build_roi_extractor(
                    recognition_roi_extractor)
                self.recognition_share_roi_extractor = False
            else:
                self.recognition_share_roi_extractor = True
                self.recognition_roi_extractor = self.mask_roi_extractor
            self.recognition_head = builder.build_recognizer(recognition_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_recognition(self):
        return hasattr(self, 'recognition_head') and self.recognition_head is not None

    def init_weights(self, pretrained=None):
        super(TextRegressE2ECascadeRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

        if self.with_recognition:  #TODO
            self.recognition_head.init_weights()
            if not self.recognition_share_roi_extractor:
                self.recognition_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_center_line=None,
                      gt_width_line=None,
                      gt_line_angle=None,
                      gt_line_length=None,
                      gt_line_x=None,
                      gt_line_y=None,
                      gt_direction=None,
                      gt_poly_pts=None,
                      gt_transcriptions=None,
                      gt_decoders=None,
                      gt_text_lengths=None,
                      proposals=None):
        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox:
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(
                    rcnn_train_cfg.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = bbox_head(bbox_feats)
            # bbox_cls_feats = bbox_roi_extractor(
            #     x[:bbox_roi_extractor.num_inputs], rois)
            # bbox_reg_feats = bbox_roi_extractor(
            #     x[:bbox_roi_extractor.num_inputs],
            #     rois,
            #     roi_scale_factor=self.reg_roi_scale_factor)
            # if self.with_shared_head:
            #     bbox_cls_feats = bbox_head(bbox_cls_feats)
            #     bbox_reg_feats = bbox_head(bbox_reg_feats)
            # cls_score, bbox_pred = bbox_head(bbox_cls_feats,
            #                                       bbox_reg_feats)

            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                                gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        # assign gts and sample proposals
        if self.with_mask:
            regress_line_bbox_assigner = build_assigner(self.train_cfg.regress_line.assigner)
            regress_line_bbox_sampler = build_sampler(
                self.train_cfg.regress_line.sampler, context=self)

            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                regress_line_assign_result = regress_line_bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                regress_line_sampling_result = regress_line_bbox_sampler.sample(
                    regress_line_assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(regress_line_sampling_result)

        # regress_line head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])

                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            para_line_pred, direction_pred = self.mask_head(mask_feats)

            center_line_targets = self.mask_head.get_target(sampling_results, gt_center_line, self.train_cfg.rcnn)
            width_line_targets = self.mask_head.get_target(sampling_results, gt_width_line, self.train_cfg.rcnn)
            control_line_angle_targets = self.mask_head.get_target(sampling_results, gt_line_angle, self.train_cfg.rcnn)
            control_line_x_targets = self.mask_head.get_target(sampling_results, gt_line_x, self.train_cfg.rcnn)
            control_line_y_targets = self.mask_head.get_target(sampling_results, gt_line_y, self.train_cfg.rcnn)
            control_line_length_targets = self.mask_head.get_target(sampling_results, gt_line_length, self.train_cfg.rcnn)
            direction_targets = self.mask_head.get_target(sampling_results, gt_direction, self.train_cfg.rcnn)
            poly_pts_targets = self.mask_head.get_target(sampling_results, gt_poly_pts, self.train_cfg.rcnn)
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
            loss_mask, control_points = self.mask_head.loss(para_line_pred, direction_pred, center_line_targets, width_line_targets,
                                            control_line_angle_targets, control_line_x_targets, control_line_y_targets,
                                            control_line_length_targets, direction_targets, poly_pts_targets, pos_rois, pos_labels)

            # print(poly_pts_targets.size())
            # print(control_points.size())
            # print("--------------------------")
            # print(poly_pts_targets)
            # print(control_points)
            # print('***************************')
            losses.update(loss_mask)
        if self.with_mask and self.with_recognition:
            if not self.recognition_share_roi_extractor:
                recognition_feats = self.recognition_roi_extractor(
                    x[:self.recognition_roi_extractor.num_inputs], pos_rois)
            else:
                recognition_feats = mask_feats
            transcription_targets = self.recognition_head.get_target(sampling_results, gt_transcriptions, self.train_cfg.rcnn)
            decoder_targets = self.recognition_head.get_target(sampling_results, gt_decoders, self.train_cfg.rcnn)
            loss_recognition = self.recognition_head(recognition_feats, word_targets=transcription_targets, decoder_targets=decoder_targets, control_points=control_points)
            losses.update(loss_recognition)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            cls_score, bbox_pred = bbox_head(bbox_feats)
            # bbox_cls_feats = bbox_roi_extractor(
            #     x[:bbox_roi_extractor.num_inputs], rois)
            # bbox_reg_feats = bbox_roi_extractor(
            #     x[:bbox_roi_extractor.num_inputs],
            #     rois,
            #     roi_scale_factor=self.reg_roi_scale_factor)
            # if self.with_shared_head:
            #     bbox_cls_feats = bbox_head(bbox_cls_feats)
            #     bbox_reg_feats = bbox_head(bbox_reg_feats)
            # cls_score, bbox_pred = bbox_head(bbox_cls_feats,
            #                                       bbox_reg_feats)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            regress_point_results, control_point = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            ms_segm_result['ensemble'] = regress_point_results

        if self.with_mask and self.with_recognition:
            recognition_results, score = self.simple_test_recogintion(x, img_meta, det_bboxes, det_labels, control_point, rescale=rescale)

        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'], recognition_results, score)
            else:
                results = ms_bbox_result['ensemble']
        else:
            if self.with_mask:
                results = {
                    stage: ms_bbox_result[stage]
                    for stage in ms_bbox_result
                }
                results = (results, ms_segm_result['ensemble'])
            else:
                results = ms_bbox_result
        return results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_roi_extractor = self.bbox_roi_extractor[i]
                bbox_head = self.bbox_head[i]

                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)

                cls_score, bbox_pred = bbox_head(bbox_feats)
                ms_scores.append(cls_score)

                if i < self.num_stages - 1:
                    bbox_label = cls_score.argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label,
                                                      bbox_pred, img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            for x, img_meta in zip(self.extract_feats(imgs), img_metas):
                regress_point_results, control_points = self.aug_test_mask(
                    x, img_meta, det_bboxes, det_labels, rescale=rescale)
                if self.with_recognition:
                    recognition_results = self.aug_test_recognition(x, img_meta, det_bboxes, det_labels, control_points, rescale=rescale)
                break

        if self.with_mask:
            return bbox_result, regress_point_results
        else:
            return bbox_result

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            result, pred_control_point = [], []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            para_regress_point, direction_pred = self.mask_head(mask_feats)
            result, pred_control_point = self.mask_head.get_seg_masks(para_regress_point, direction_pred, _bboxes, det_labels,
                                                  self.test_cfg.rcnn, ori_shape, scale_factor, rescale)

        return result, pred_control_point

    def aug_test_mask(self, x, img_meta, det_bboxes, det_labels, rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        img_shape = img_meta[0]['img_shape']
        flip = img_meta[0]['flip']
        if det_bboxes.shape[0] == 0:
            result, pred_control_point = [], []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            para_regress_point, direction_pred = self.mask_head(mask_feats)
            result, pred_control_point = self.mask_head.get_seg_masks(para_regress_point, direction_pred, _bboxes, det_labels,
                                                  self.test_cfg.rcnn, ori_shape, scale_factor, rescale)
        return result, pred_control_point

    def simple_test_recogintion(self, x, img_meta, det_bboxes, det_labels, control_points, rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            result, score = [], []
        else:
            _bboxes = (det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            recognition_rois = bbox2roi([_bboxes])
            recognition_feats = self.recognition_roi_extractor(x[:len(self.recognition_roi_extractor.featmap_strides)], recognition_rois)
            result, score, _ = self.recognition_head(recognition_feats, control_points=control_points, use_beam_search=False)
        return result, score

    def aug_test_recognition(self, x, img_meta, det_bboxes, det_labels, control_points, rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        img_shape = img_meta[0]['img_shape']
        flip = img_meta[0]['flip']
        if det_bboxes.shape[0] == 0:
            result, score = [], []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip)
            mask_rois = bbox2roi([_bboxes])
            recognition_rois = bbox2roi([_bboxes])
            recognition_feats = self.recognition_roi_extractor(x[:len(self.recognition_roi_extractor.featmap_strides)], recognition_rois)
            result, score, _ = self.recognition_head(recognition_feats, control_points=control_points, use_beam_search=False)
        return result, score
