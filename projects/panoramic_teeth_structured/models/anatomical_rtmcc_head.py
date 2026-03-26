from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData, PixelData
from torch import Tensor, nn

from mmpose.models.heads.base_head import BaseHead
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList


def _stack_field(batch_data_samples: OptSampleList, field_name: str) -> Tensor:
    return torch.stack(
        [getattr(sample.gt_fields, field_name) for sample in batch_data_samples],
        dim=0).float()


def _stack_label(batch_data_samples: OptSampleList, label_name: str) -> Tensor:
    value = torch.stack([
        getattr(sample.gt_instance_labels, label_name)
        for sample in batch_data_samples
    ],
                        dim=0).float()
    if value.ndim >= 2 and value.shape[1] == 1:
        value = value.squeeze(1)
    return value


@MODELS.register_module()
class AnatomicalRTMCCHead(BaseHead):
    """Anatomy-aware point head with structured auxiliary supervision."""

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 input_size: Tuple[int, int],
                 contour_points: int = 16,
                 feat_channels: int = 256,
                 num_convs: int = 2,
                 contour_hidden_dim: int = 512,
                 point_head_cfg: OptConfigType = None,
                 root_bce_weight: float = 1.0,
                 root_dice_weight: float = 1.0,
                 boundary_bce_weight: float = 1.0,
                 boundary_dice_weight: float = 1.0,
                 contour_weight: float = 2.0,
                 contour_attach_weight: float = 0.5,
                 mesial_point_weight: float = 1.0,
                 apex_point_weight: float = 1.0,
                 distal_point_weight: float = 1.0,
                 side_attach_weight: float = 0.5,
                 side_repel_weight: float = 0.2,
                 point_gap_weight: float = 0.2,
                 vertical_order_weight: float = 0.2,
                 apex_consistency_weight: float = 0.5,
                 repel_margin: float = 0.03,
                 point_gap_margin: float = 0.02,
                 vertical_order_margin: float = 0.01,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support multiple inputs.')

        self.input_size = tuple(int(v) for v in input_size)
        self.contour_points = int(contour_points)

        self.root_bce_weight = float(root_bce_weight)
        self.root_dice_weight = float(root_dice_weight)
        self.boundary_bce_weight = float(boundary_bce_weight)
        self.boundary_dice_weight = float(boundary_dice_weight)
        self.contour_weight = float(contour_weight)
        self.contour_attach_weight = float(contour_attach_weight)
        self.mesial_point_weight = float(mesial_point_weight)
        self.apex_point_weight = float(apex_point_weight)
        self.distal_point_weight = float(distal_point_weight)
        self.side_attach_weight = float(side_attach_weight)
        self.side_repel_weight = float(side_repel_weight)
        self.point_gap_weight = float(point_gap_weight)
        self.vertical_order_weight = float(vertical_order_weight)
        self.apex_consistency_weight = float(apex_consistency_weight)
        self.repel_margin = float(repel_margin)
        self.point_gap_margin = float(point_gap_margin)
        self.vertical_order_margin = float(vertical_order_margin)

        layers = []
        current_channels = int(in_channels)
        for _ in range(int(num_convs)):
            layers.extend([
                nn.Conv2d(current_channels, feat_channels, 3, padding=1),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = feat_channels
        self.decoder = nn.Sequential(*layers)
        self.structure_classifier = nn.Conv2d(current_channels, 3, kernel_size=1)
        self.contour_mlp = nn.Sequential(
            nn.Linear(current_channels * 2, contour_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(contour_hidden_dim, 2 * self.contour_points * 2))

        if point_head_cfg is None:
            raise ValueError('`point_head_cfg` is required.')

        point_head_cfg = point_head_cfg.copy()
        point_head_cfg.setdefault('type', 'RTMCCHead')
        point_head_cfg.setdefault('in_channels', int(in_channels))
        point_head_cfg.setdefault('input_size', self.input_size)

        apex_cfg = point_head_cfg.copy()
        apex_cfg['out_channels'] = 1
        mesial_cfg = point_head_cfg.copy()
        mesial_cfg['out_channels'] = 2
        distal_cfg = point_head_cfg.copy()
        distal_cfg['out_channels'] = 2

        self.apex_head = MODELS.build(apex_cfg)
        self.mesial_head = MODELS.build(mesial_cfg)
        self.distal_head = MODELS.build(distal_cfg)

        self.simcc_split_ratio = float(self.apex_head.simcc_split_ratio)

    def _structure_forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        x = feats[-1]
        x = self.decoder(x)

        structure_logits = self.structure_classifier(x)
        output_size = (int(self.input_size[1]), int(self.input_size[0]))
        structure_logits = F.interpolate(
            structure_logits,
            size=output_size,
            mode='bilinear',
            align_corners=False)

        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)
        contour = self.contour_mlp(torch.cat([avg, max_pool], dim=1))
        contour = contour.view(-1, 2, self.contour_points, 2).sigmoid()

        width = max(self.input_size[0] - 1, 1)
        height = max(self.input_size[1] - 1, 1)
        contour = torch.cat(
            [contour[..., 0:1] * width, contour[..., 1:2] * height], dim=-1)
        return structure_logits, contour

    def _point_forward(self, feats: Tuple[Tensor]) -> Dict[str, Tuple[Tensor, Tensor]]:
        return dict(
            mesial=self.mesial_head.forward(feats),
            apex=self.apex_head.forward(feats),
            distal=self.distal_head.forward(feats))

    @staticmethod
    def _dice_loss(logits: Tensor, targets: Tensor,
                   eps: float = 1e-6) -> Tensor:
        probs = logits.sigmoid().flatten(2)
        targets = targets.flatten(2)
        intersection = (probs * targets).sum(dim=2)
        denominator = probs.sum(dim=2) + targets.sum(dim=2)
        dice = (2 * intersection + eps) / (denominator + eps)
        return 1.0 - dice.mean()

    @staticmethod
    def _sample_maps(maps: Tensor, points: Tensor) -> Tensor:
        height, width = maps.shape[-2:]
        x = points[..., 0] / max(width - 1, 1) * 2 - 1
        y = points[..., 1] / max(height - 1, 1) * 2 - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            maps,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2)

    def _soft_argmax(self, pred_x: Tensor, pred_y: Tensor) -> Tensor:
        x_prob = pred_x.softmax(dim=-1)
        y_prob = pred_y.softmax(dim=-1)
        x_range = torch.arange(
            pred_x.shape[-1], device=pred_x.device, dtype=pred_x.dtype)
        y_range = torch.arange(
            pred_y.shape[-1], device=pred_y.device, dtype=pred_y.dtype)
        x_coord = (x_prob * x_range.view(1, 1, -1)).sum(dim=-1)
        y_coord = (y_prob * y_range.view(1, 1, -1)).sum(dim=-1)
        coords = torch.stack([x_coord, y_coord], dim=-1)
        coords = coords / self.simcc_split_ratio
        return coords

    def _decode_points(self,
                       point_logits: Dict[str, Tuple[Tensor, Tensor]]
                       ) -> Tuple[list, list, list]:
        mesial_preds = self.mesial_head.decode(point_logits['mesial'])
        apex_preds = self.apex_head.decode(point_logits['apex'])
        distal_preds = self.distal_head.decode(point_logits['distal'])
        return mesial_preds, apex_preds, distal_preds

    def _merge_tta_logits(self, feats, batch_data_samples):
        del batch_data_samples
        assert isinstance(feats, list) and len(feats) == 2

        structure_orig, contours_orig = self._structure_forward(feats[0])
        structure_flip, contours_flip = self._structure_forward(feats[1])
        structure_flip = structure_flip.flip(-1)
        structure_flip = structure_flip[:, [0, 2, 1]]
        contours_flip = contours_flip.clone()
        contours_flip[..., 0] = (self.input_size[0] - 1) - contours_flip[..., 0]
        contours_flip = contours_flip[:, [1, 0]]

        point_orig = self._point_forward(feats[0])
        point_flip = self._point_forward(feats[1])

        apex_flip = flip_vectors(
            point_flip['apex'][0], point_flip['apex'][1], flip_indices=[0])
        mesial_from_flip = flip_vectors(
            point_flip['distal'][0], point_flip['distal'][1], flip_indices=[1, 0])
        distal_from_flip = flip_vectors(
            point_flip['mesial'][0], point_flip['mesial'][1], flip_indices=[1, 0])

        point_logits = dict(
            mesial=((point_orig['mesial'][0] + mesial_from_flip[0]) * 0.5,
                    (point_orig['mesial'][1] + mesial_from_flip[1]) * 0.5),
            apex=((point_orig['apex'][0] + apex_flip[0]) * 0.5,
                  (point_orig['apex'][1] + apex_flip[1]) * 0.5),
            distal=((point_orig['distal'][0] + distal_from_flip[0]) * 0.5,
                    (point_orig['distal'][1] + distal_from_flip[1]) * 0.5))

        structure_logits = 0.5 * (structure_orig + structure_flip)
        contours = 0.5 * (contours_orig + contours_flip)
        return structure_logits, contours, point_logits

    def forward(self, feats: Tuple[Tensor]):
        structure_logits, contours = self._structure_forward(feats)
        point_logits = self._point_forward(feats)
        return structure_logits, contours, point_logits

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        del train_cfg
        structure_logits, contours, point_logits = self.forward(feats)

        gt_root = _stack_field(batch_data_samples, 'root_mask').to(
            structure_logits.device)
        gt_mesial = _stack_field(batch_data_samples, 'mesial_boundary').to(
            structure_logits.device)
        gt_distal = _stack_field(batch_data_samples, 'distal_boundary').to(
            structure_logits.device)
        gt_mesial_dt = _stack_field(batch_data_samples, 'mesial_distance').to(
            structure_logits.device)
        gt_distal_dt = _stack_field(batch_data_samples, 'distal_distance').to(
            structure_logits.device)
        gt_mesial_anatomy_dt = _stack_field(
            batch_data_samples, 'mesial_anatomy_distance').to(structure_logits.device)
        gt_distal_anatomy_dt = _stack_field(
            batch_data_samples, 'distal_anatomy_distance').to(structure_logits.device)

        gt_mesial_contour = _stack_label(batch_data_samples, 'mesial_contour').to(
            contours.device)
        gt_distal_contour = _stack_label(batch_data_samples, 'distal_contour').to(
            contours.device)
        gt_apex_midpoint = _stack_label(
            batch_data_samples, 'apex_midpoint_target').to(contours.device)

        root_logits = structure_logits[:, 0:1]
        boundary_logits = structure_logits[:, 1:3]
        boundary_targets = torch.cat([gt_mesial, gt_distal], dim=1)

        width = max(self.input_size[0] - 1, 1)
        height = max(self.input_size[1] - 1, 1)
        norm = contours.new_tensor([width, height]).view(1, 1, 1, 2)
        pred_contours_norm = contours / norm
        gt_contours = torch.stack([gt_mesial_contour, gt_distal_contour], dim=1)
        gt_contours_norm = gt_contours / norm

        contour_loss = F.smooth_l1_loss(pred_contours_norm, gt_contours_norm)
        contour_attach = 0.5 * (
            self._sample_maps(gt_mesial_dt, contours[:, 0]).mean() +
            self._sample_maps(gt_distal_dt, contours[:, 1]).mean())

        mesial_x = _stack_label(batch_data_samples,
                                'mesial_keypoint_x_labels').to(contours.device)
        mesial_y = _stack_label(batch_data_samples,
                                'mesial_keypoint_y_labels').to(contours.device)
        mesial_w = _stack_label(batch_data_samples,
                                'mesial_keypoint_weights').to(contours.device)
        apex_x = _stack_label(batch_data_samples,
                              'apex_keypoint_x_labels').to(contours.device)
        apex_y = _stack_label(batch_data_samples,
                              'apex_keypoint_y_labels').to(contours.device)
        apex_w = _stack_label(batch_data_samples,
                              'apex_keypoint_weights').to(contours.device)
        distal_x = _stack_label(batch_data_samples,
                                'distal_keypoint_x_labels').to(contours.device)
        distal_y = _stack_label(batch_data_samples,
                                'distal_keypoint_y_labels').to(contours.device)
        distal_w = _stack_label(batch_data_samples,
                                'distal_keypoint_weights').to(contours.device)

        mesial_point_loss = self.mesial_head.loss_module(
            point_logits['mesial'], (mesial_x, mesial_y), mesial_w)
        apex_point_loss = self.apex_head.loss_module(
            point_logits['apex'], (apex_x, apex_y), apex_w)
        distal_point_loss = self.distal_head.loss_module(
            point_logits['distal'], (distal_x, distal_y), distal_w)

        mesial_coords = self._soft_argmax(*point_logits['mesial'])
        apex_coords = self._soft_argmax(*point_logits['apex'])
        distal_coords = self._soft_argmax(*point_logits['distal'])
        keypoints = torch.cat([mesial_coords, apex_coords, distal_coords], dim=1)
        keypoints_norm = keypoints / keypoints.new_tensor(
            [width, height]).view(1, 1, 2)

        mesial_attach = self._sample_maps(
            gt_mesial_anatomy_dt, keypoints[:, [0, 1]]).mean()
        distal_attach = self._sample_maps(
            gt_distal_anatomy_dt, keypoints[:, [3, 4]]).mean()
        side_attach = 0.5 * (mesial_attach + distal_attach)

        mesial_repel = F.relu(
            self.repel_margin -
            self._sample_maps(gt_distal_anatomy_dt, keypoints[:, [0, 1]])).mean()
        distal_repel = F.relu(
            self.repel_margin -
            self._sample_maps(gt_mesial_anatomy_dt, keypoints[:, [3, 4]])).mean()
        side_repel = 0.5 * (mesial_repel + distal_repel)

        mesial_gap = torch.norm(
            keypoints_norm[:, 0] - keypoints_norm[:, 1], dim=-1)
        distal_gap = torch.norm(
            keypoints_norm[:, 3] - keypoints_norm[:, 4], dim=-1)
        point_gap = 0.5 * (
            F.relu(self.point_gap_margin - mesial_gap).mean() +
            F.relu(self.point_gap_margin - distal_gap).mean())

        y_mc = keypoints_norm[:, 0, 1]
        y_mb = keypoints_norm[:, 1, 1]
        y_a = keypoints_norm[:, 2, 1]
        y_db = keypoints_norm[:, 3, 1]
        y_dc = keypoints_norm[:, 4, 1]
        vertical_order = (
            F.relu(y_mc + self.vertical_order_margin - y_mb).mean() +
            F.relu(y_dc + self.vertical_order_margin - y_db).mean() +
            F.relu(y_mb + self.vertical_order_margin - y_a).mean() +
            F.relu(y_db + self.vertical_order_margin - y_a).mean()) * 0.25

        pred_contour_apex = 0.5 * (contours[:, 0, -1] + contours[:, 1, -1])
        pred_apex = keypoints[:, 2]
        pred_apex_norm = pred_apex / pred_apex.new_tensor([width, height]).view(
            1, 2)
        gt_apex_midpoint_norm = gt_apex_midpoint / gt_apex_midpoint.new_tensor(
            [width, height]).view(1, 2)
        pred_contour_apex_norm = pred_contour_apex / pred_contour_apex.new_tensor(
            [width, height]).view(1, 2)
        apex_consistency = 0.5 * F.smooth_l1_loss(
            pred_apex_norm, gt_apex_midpoint_norm) + 0.5 * F.smooth_l1_loss(
                pred_apex_norm, pred_contour_apex_norm)

        return {
            'loss_root_bce':
            F.binary_cross_entropy_with_logits(root_logits, gt_root) *
            self.root_bce_weight,
            'loss_root_dice':
            self._dice_loss(root_logits, gt_root) * self.root_dice_weight,
            'loss_boundary_bce':
            F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
            * self.boundary_bce_weight,
            'loss_boundary_dice':
            self._dice_loss(boundary_logits, boundary_targets) *
            self.boundary_dice_weight,
            'loss_contour':
            contour_loss * self.contour_weight,
            'loss_contour_attach':
            contour_attach * self.contour_attach_weight,
            'loss_kpt_mesial':
            mesial_point_loss * self.mesial_point_weight,
            'loss_kpt_apex':
            apex_point_loss * self.apex_point_weight,
            'loss_kpt_distal':
            distal_point_loss * self.distal_point_weight,
            'loss_side_attach':
            side_attach * self.side_attach_weight,
            'loss_side_repel':
            side_repel * self.side_repel_weight,
            'loss_point_gap':
            point_gap * self.point_gap_weight,
            'loss_vertical_order':
            vertical_order * self.vertical_order_weight,
            'loss_apex_consistency':
            apex_consistency * self.apex_consistency_weight,
        }

    def predict(self,
                feats,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}):
        if test_cfg.get('flip_test', False):
            structure_logits, contours, point_logits = self._merge_tta_logits(
                feats, batch_data_samples)
        else:
            structure_logits, contours, point_logits = self.forward(feats)

        mesial_preds, apex_preds, distal_preds = self._decode_points(point_logits)
        structure_probs = structure_logits.sigmoid()

        pred_instances = []
        pred_fields = []
        for index, (mesial_pred, apex_pred,
                    distal_pred) in enumerate(zip(mesial_preds, apex_preds,
                                                  distal_preds)):
            keypoints = np.concatenate(
                [mesial_pred.keypoints, apex_pred.keypoints, distal_pred.keypoints],
                axis=1)
            keypoint_scores = np.concatenate([
                mesial_pred.keypoint_scores,
                apex_pred.keypoint_scores,
                distal_pred.keypoint_scores
            ],
                                           axis=1)
            mesial_np = contours[index:index + 1, 0].detach().cpu().numpy()
            distal_np = contours[index:index + 1, 1].detach().cpu().numpy()
            pred_instances.append(
                InstanceData(
                    keypoints=keypoints,
                    keypoint_scores=keypoint_scores,
                    mesial_contour=mesial_np,
                    distal_contour=distal_np))
            pred_fields.append(
                PixelData(
                    root_mask=structure_probs[index, 0].detach(),
                    mesial_boundary=structure_probs[index, 1].detach(),
                    distal_boundary=structure_probs[index, 2].detach()))

        return pred_instances, pred_fields
