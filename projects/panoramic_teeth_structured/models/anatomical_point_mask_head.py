from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

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
class AnatomicalPointMaskHead(BaseHead):
    """Predict anatomical points with root-mask auxiliary supervision.

    The default mode keeps the existing point+mask behavior. When
    ``side_prior_mode='anatomy'`` and contour auxiliary weights are enabled,
    the head borrows the old structured baseline's dense side-contour prior
    without exposing extra outputs at inference time.
    """

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 input_size: Tuple[int, int],
                 feat_channels: int = 256,
                 num_convs: int = 2,
                 contour_points: int = 16,
                 contour_hidden_dim: int = 512,
                 point_head_cfg: OptConfigType = None,
                 root_bce_weight: float = 1.0,
                 root_dice_weight: float = 1.0,
                 mesial_point_weight: float = 1.0,
                 apex_point_weight: float = 1.0,
                 distal_point_weight: float = 1.0,
                 side_prior_mode: str = 'polyline',
                 contour_aux_weight: float = 0.0,
                 contour_attach_aux_weight: float = 0.0,
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

        if side_prior_mode not in {'polyline', 'anatomy'}:
            raise ValueError(
                f'Unsupported side_prior_mode={side_prior_mode!r}.')

        self.input_size = tuple(int(v) for v in input_size)
        self.contour_points = int(contour_points)
        self.side_prior_mode = side_prior_mode

        self.root_bce_weight = float(root_bce_weight)
        self.root_dice_weight = float(root_dice_weight)
        self.mesial_point_weight = float(mesial_point_weight)
        self.apex_point_weight = float(apex_point_weight)
        self.distal_point_weight = float(distal_point_weight)
        self.contour_aux_weight = float(contour_aux_weight)
        self.contour_attach_aux_weight = float(contour_attach_aux_weight)
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
        self.root_classifier = nn.Conv2d(current_channels, 1, kernel_size=1)

        self.enable_contour_aux = (
            self.contour_aux_weight > 0 or self.contour_attach_aux_weight > 0)
        if self.enable_contour_aux:
            self.contour_mlp = nn.Sequential(
                nn.Linear(current_channels * 2, contour_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(contour_hidden_dim, 2 * self.contour_points * 2))
        else:
            self.contour_mlp = None

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

    def _decode_features(self, feats: Tuple[Tensor]) -> Tensor:
        return self.decoder(feats[-1])

    def _root_logits_from_decoded(self, decoded: Tensor) -> Tensor:
        root_logits = self.root_classifier(decoded)
        output_size = (int(self.input_size[1]), int(self.input_size[0]))
        return F.interpolate(
            root_logits,
            size=output_size,
            mode='bilinear',
            align_corners=False)

    def _root_forward(self, feats: Tuple[Tensor]) -> Tensor:
        return self._root_logits_from_decoded(self._decode_features(feats))

    def _contour_forward_from_decoded(self, decoded: Tensor) -> Optional[Tensor]:
        if not self.enable_contour_aux:
            return None

        avg = F.adaptive_avg_pool2d(decoded, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(decoded, 1).flatten(1)
        contour = self.contour_mlp(torch.cat([avg, max_pool], dim=1))
        contour = contour.view(-1, 2, self.contour_points, 2).sigmoid()

        width = max(self.input_size[0] - 1, 1)
        height = max(self.input_size[1] - 1, 1)
        return torch.cat(
            [contour[..., 0:1] * width, contour[..., 1:2] * height], dim=-1)

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
        return coords / self.simcc_split_ratio

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

        root_orig = self._root_forward(feats[0])
        root_flip = self._root_forward(feats[1]).flip(-1)

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
        root_logits = 0.5 * (root_orig + root_flip)
        return root_logits, point_logits

    def forward(self, feats: Tuple[Tensor]):
        decoded = self._decode_features(feats)
        root_logits = self._root_logits_from_decoded(decoded)
        point_logits = self._point_forward(feats)
        contour_preds = self._contour_forward_from_decoded(decoded)
        return root_logits, point_logits, contour_preds

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        del train_cfg
        root_logits, point_logits, contour_preds = self.forward(feats)

        gt_root = _stack_field(batch_data_samples, 'root_mask').to(root_logits.device)
        if self.side_prior_mode == 'anatomy':
            gt_mesial_dt = _stack_field(
                batch_data_samples, 'mesial_anatomy_distance').to(
                    root_logits.device)
            gt_distal_dt = _stack_field(
                batch_data_samples, 'distal_anatomy_distance').to(
                    root_logits.device)
        else:
            gt_mesial_dt = _stack_field(
                batch_data_samples, 'mesial_polyline_distance').to(
                    root_logits.device)
            gt_distal_dt = _stack_field(
                batch_data_samples, 'distal_polyline_distance').to(
                    root_logits.device)
        gt_apex_midpoint = _stack_label(
            batch_data_samples, 'apex_midpoint_target').to(root_logits.device)

        mesial_x = _stack_label(batch_data_samples,
                                'mesial_keypoint_x_labels').to(root_logits.device)
        mesial_y = _stack_label(batch_data_samples,
                                'mesial_keypoint_y_labels').to(root_logits.device)
        mesial_w = _stack_label(batch_data_samples,
                                'mesial_keypoint_weights').to(root_logits.device)
        apex_x = _stack_label(batch_data_samples,
                              'apex_keypoint_x_labels').to(root_logits.device)
        apex_y = _stack_label(batch_data_samples,
                              'apex_keypoint_y_labels').to(root_logits.device)
        apex_w = _stack_label(batch_data_samples,
                              'apex_keypoint_weights').to(root_logits.device)
        distal_x = _stack_label(batch_data_samples,
                                'distal_keypoint_x_labels').to(root_logits.device)
        distal_y = _stack_label(batch_data_samples,
                                'distal_keypoint_y_labels').to(root_logits.device)
        distal_w = _stack_label(batch_data_samples,
                                'distal_keypoint_weights').to(root_logits.device)

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

        width = max(self.input_size[0] - 1, 1)
        height = max(self.input_size[1] - 1, 1)
        keypoints_norm = keypoints / keypoints.new_tensor(
            [width, height]).view(1, 1, 2)

        losses = {
            'loss_root_bce':
            F.binary_cross_entropy_with_logits(root_logits, gt_root) *
            self.root_bce_weight,
            'loss_root_dice':
            self._dice_loss(root_logits, gt_root) * self.root_dice_weight,
            'loss_kpt_mesial':
            mesial_point_loss * self.mesial_point_weight,
            'loss_kpt_apex':
            apex_point_loss * self.apex_point_weight,
            'loss_kpt_distal':
            distal_point_loss * self.distal_point_weight,
        }

        if self.enable_contour_aux and contour_preds is not None:
            gt_mesial_contour = _stack_label(batch_data_samples,
                                             'mesial_contour').to(contour_preds.device)
            gt_distal_contour = _stack_label(batch_data_samples,
                                             'distal_contour').to(contour_preds.device)
            gt_mesial_side_dt = _stack_field(batch_data_samples, 'mesial_distance').to(
                contour_preds.device)
            gt_distal_side_dt = _stack_field(batch_data_samples, 'distal_distance').to(
                contour_preds.device)

            norm = contour_preds.new_tensor([width, height]).view(1, 1, 1, 2)
            gt_contours = torch.stack([gt_mesial_contour, gt_distal_contour], dim=1)
            contour_loss = F.smooth_l1_loss(contour_preds / norm, gt_contours / norm)
            contour_attach = 0.5 * (
                self._sample_maps(gt_mesial_side_dt, contour_preds[:, 0]).mean() +
                self._sample_maps(gt_distal_side_dt, contour_preds[:, 1]).mean())
            losses.update(
                loss_contour_aux=contour_loss * self.contour_aux_weight,
                loss_contour_attach_aux=contour_attach *
                self.contour_attach_aux_weight)

        mesial_attach = self._sample_maps(
            gt_mesial_dt, keypoints[:, [0, 1, 2]]).mean()
        distal_attach = self._sample_maps(
            gt_distal_dt, keypoints[:, [2, 3, 4]]).mean()
        side_attach = 0.5 * (mesial_attach + distal_attach)

        mesial_repel = F.relu(
            self.repel_margin -
            self._sample_maps(gt_distal_dt, keypoints[:, [0, 1]])).mean()
        distal_repel = F.relu(
            self.repel_margin -
            self._sample_maps(gt_mesial_dt, keypoints[:, [3, 4]])).mean()
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

        pred_apex_norm = keypoints[:, 2] / keypoints.new_tensor([width, height]).view(
            1, 2)
        gt_apex_midpoint_norm = gt_apex_midpoint / gt_apex_midpoint.new_tensor(
            [width, height]).view(1, 2)
        apex_consistency = F.smooth_l1_loss(pred_apex_norm,
                                            gt_apex_midpoint_norm)

        losses.update(
            loss_side_attach=side_attach * self.side_attach_weight,
            loss_side_repel=side_repel * self.side_repel_weight,
            loss_point_gap=point_gap * self.point_gap_weight,
            loss_vertical_order=vertical_order * self.vertical_order_weight,
            loss_apex_consistency=apex_consistency *
            self.apex_consistency_weight)
        return losses

    def predict(self,
                feats,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}):
        if test_cfg.get('flip_test', False):
            root_logits, point_logits = self._merge_tta_logits(
                feats, batch_data_samples)
        else:
            root_logits, point_logits, _ = self.forward(feats)

        mesial_preds, apex_preds, distal_preds = self._decode_points(point_logits)
        root_probs = root_logits.sigmoid()

        pred_instances = []
        pred_fields = []
        for index, (mesial_pred, apex_pred,
                    distal_pred) in enumerate(zip(mesial_preds, apex_preds,
                                                  distal_preds)):
            keypoints = torch.cat([
                torch.from_numpy(mesial_pred.keypoints),
                torch.from_numpy(apex_pred.keypoints),
                torch.from_numpy(distal_pred.keypoints)
            ],
                                  dim=1).cpu().numpy()
            keypoint_scores = torch.cat([
                torch.from_numpy(mesial_pred.keypoint_scores),
                torch.from_numpy(apex_pred.keypoint_scores),
                torch.from_numpy(distal_pred.keypoint_scores)
            ],
                                        dim=1).cpu().numpy()
            pred_instances.append(
                InstanceData(
                    keypoints=keypoints, keypoint_scores=keypoint_scores))
            pred_fields.append(
                PixelData(root_mask=root_probs[index, 0].detach()))

        return pred_instances, pred_fields
