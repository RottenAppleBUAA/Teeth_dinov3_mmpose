from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData, PixelData
from torch import Tensor, nn

from mmpose.models.heads.base_head import BaseHead
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
class StructuredContourHead(BaseHead):
    """Predict root/side structure maps and reconstruct keypoints from
    anatomical side contours.
    """

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 input_size: Tuple[int, int],
                 contour_points: int = 16,
                 feat_channels: int = 256,
                 num_convs: int = 2,
                 contour_hidden_dim: int = 512,
                 row_sigma: float = 6.0,
                 contour_temperature: float = 20.0,
                 root_bce_weight: float = 1.0,
                 root_dice_weight: float = 1.0,
                 boundary_bce_weight: float = 1.0,
                 boundary_dice_weight: float = 1.0,
                 contour_weight: float = 2.0,
                 recon_kpt_weight: float = 2.0,
                 attach_weight: float = 0.5,
                 order_weight: float = 0.2,
                 apex_weight: float = 0.5,
                 order_margin: float = 0.02,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support multiple inputs.')

        self.input_size = tuple(int(v) for v in input_size)
        self.contour_points = int(contour_points)
        self.order_margin = float(order_margin)
        self.row_sigma = float(row_sigma)
        self.contour_temperature = float(contour_temperature)

        self.root_bce_weight = float(root_bce_weight)
        self.root_dice_weight = float(root_dice_weight)
        self.boundary_bce_weight = float(boundary_bce_weight)
        self.boundary_dice_weight = float(boundary_dice_weight)
        self.contour_weight = float(contour_weight)
        self.recon_kpt_weight = float(recon_kpt_weight)
        self.attach_weight = float(attach_weight)
        self.order_weight = float(order_weight)
        self.apex_weight = float(apex_weight)

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
        self.depth_mlp = nn.Sequential(
            nn.Linear(current_channels * 2, contour_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(contour_hidden_dim, self.contour_points + 1))

    def _decode_depth_anchors(self, pooled: Tensor, height: int) -> Tensor:
        depth_params = self.depth_mlp(pooled)
        top_raw = depth_params[:, 0]
        span_raw = depth_params[:, 1]
        seg_raw = depth_params[:, 2:]

        max_y = float(max(height - 1, 1))
        top = torch.sigmoid(top_raw) * max_y
        span = torch.sigmoid(span_raw) * (max_y - top)

        segments = F.softplus(seg_raw) + 1e-4
        cumulative = torch.cumsum(segments, dim=1)
        cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-6)
        start = torch.zeros_like(top[:, None])
        return top[:, None] + span[:, None] * torch.cat([start, cumulative], dim=1)

    def _decode_contours_from_structure(self, structure_logits: Tensor,
                                        pooled: Tensor) -> Tensor:
        structure_probs = structure_logits.sigmoid()
        root_gate = 0.5 + 0.5 * structure_probs[:, 0:1]
        boundary_probs = structure_probs[:, 1:3] * root_gate

        _, _, height, width = boundary_probs.shape
        y_anchors = self._decode_depth_anchors(pooled, height)

        y_coords = torch.arange(
            height, device=boundary_probs.device,
            dtype=boundary_probs.dtype).view(1, 1, height)
        x_coords = torch.arange(
            width, device=boundary_probs.device,
            dtype=boundary_probs.dtype).view(1, 1, width)

        row_sigma = max(self.row_sigma, 1e-3)
        row_prior = torch.exp(-0.5 * ((y_coords - y_anchors.unsqueeze(-1)) /
                                      row_sigma)**2)
        row_prior = row_prior / row_prior.sum(dim=-1,
                                              keepdim=True).clamp_min(1e-6)

        contours = []
        for side_idx in range(2):
            side_response = boundary_probs[:, side_idx].unsqueeze(1)
            point_response = side_response * row_prior.unsqueeze(-1)

            x_scores = point_response.sum(dim=2)
            x_probs = F.softmax(x_scores * self.contour_temperature, dim=-1)
            x = (x_probs * x_coords).sum(dim=-1)

            y_scores = point_response.sum(dim=3) + row_prior * 1e-3
            y_probs = y_scores / y_scores.sum(dim=-1, keepdim=True).clamp_min(
                1e-6)
            y = (y_probs * y_coords).sum(dim=-1)

            contours.append(torch.stack([x, y], dim=-1))

        return torch.stack(contours, dim=1)

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
        pooled = torch.cat([avg, max_pool], dim=1)
        contour = self._decode_contours_from_structure(structure_logits, pooled)
        return structure_logits, contour

    @staticmethod
    def _derive_keypoints(contours: Tensor) -> Tensor:
        mesial = contours[:, 0]
        distal = contours[:, 1]
        apex = 0.5 * (mesial[:, -1] + distal[:, -1])
        return torch.stack([
            mesial[:, 0],
            mesial[:, 1],
            apex,
            distal[:, 1],
            distal[:, 0],
        ],
                           dim=1)

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

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        structure_logits, contours = self._structure_forward(feats)
        keypoints = self._derive_keypoints(contours)
        return structure_logits, contours, keypoints

    def _merge_tta(self, orig_outputs, flip_outputs):
        structure_logits, contours, _ = orig_outputs
        structure_logits_flip, contours_flip, _ = flip_outputs

        structure_logits_flip = structure_logits_flip.flip(-1)
        structure_logits_flip = structure_logits_flip[:, [0, 2, 1]]

        contours_flip = contours_flip.clone()
        contours_flip[..., 0] = (self.input_size[0] - 1) - contours_flip[..., 0]
        contours_flip = contours_flip[:, [1, 0]]

        merged_logits = 0.5 * (structure_logits + structure_logits_flip)
        merged_contours = 0.5 * (contours + contours_flip)
        merged_keypoints = self._derive_keypoints(merged_contours)
        return merged_logits, merged_contours, merged_keypoints

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        del train_cfg
        structure_logits, contours, keypoints = self.forward(feats)

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

        gt_mesial_contour = _stack_label(batch_data_samples, 'mesial_contour').to(
            contours.device)
        gt_distal_contour = _stack_label(batch_data_samples, 'distal_contour').to(
            contours.device)
        gt_keypoints = _stack_label(batch_data_samples, 'keypoint_targets').to(
            keypoints.device)
        gt_keypoint_weights = _stack_label(
            batch_data_samples, 'keypoint_weights').to(keypoints.device)
        gt_apex = _stack_label(batch_data_samples, 'apex_target').to(
            keypoints.device)

        root_logits = structure_logits[:, 0:1]
        boundary_logits = structure_logits[:, 1:3]
        boundary_targets = torch.cat([gt_mesial, gt_distal], dim=1)

        width = max(self.input_size[0] - 1, 1)
        height = max(self.input_size[1] - 1, 1)
        norm = contours.new_tensor([width, height]).view(1, 1, 1, 2)
        pred_contours_norm = contours / norm
        gt_contours = torch.stack([gt_mesial_contour, gt_distal_contour], dim=1)
        gt_contours_norm = gt_contours / norm

        gt_keypoints_norm = gt_keypoints / keypoints.new_tensor(
            [width, height]).view(1, 1, 2)
        pred_keypoints_norm = keypoints / keypoints.new_tensor(
            [width, height]).view(1, 1, 2)
        gt_apex_norm = gt_apex / gt_apex.new_tensor([width, height]).view(1, 2)
        pred_apex = keypoints[:, 2]
        pred_apex_norm = pred_keypoints_norm[:, 2]

        contour_loss = F.smooth_l1_loss(pred_contours_norm, gt_contours_norm)
        keypoint_l1 = F.smooth_l1_loss(
            pred_keypoints_norm, gt_keypoints_norm, reduction='none').sum(dim=-1)
        keypoint_loss = (keypoint_l1 * gt_keypoint_weights).sum() / (
            gt_keypoint_weights.sum().clamp_min(1.0))

        mesial_attach = self._sample_maps(gt_mesial_dt, contours[:, 0]).mean()
        distal_attach = self._sample_maps(gt_distal_dt, contours[:, 1]).mean()
        mesial_key_attach = self._sample_maps(gt_mesial_dt,
                                              keypoints[:, [0, 1]]).mean()
        distal_key_attach = self._sample_maps(gt_distal_dt,
                                              keypoints[:, [3, 4]]).mean()
        attach_loss = (mesial_attach + distal_attach + mesial_key_attach +
                       distal_key_attach) * 0.25

        dist_mc = torch.norm(keypoints[:, 0] - keypoints[:, 2], dim=-1)
        dist_mb = torch.norm(keypoints[:, 1] - keypoints[:, 2], dim=-1)
        dist_dc = torch.norm(keypoints[:, 4] - keypoints[:, 2], dim=-1)
        dist_db = torch.norm(keypoints[:, 3] - keypoints[:, 2], dim=-1)
        margin = self.order_margin
        order_loss = (
            F.relu(dist_mb - dist_mc + margin).mean() +
            F.relu(dist_db - dist_dc + margin).mean()) * 0.5

        apex_reg = F.smooth_l1_loss(pred_apex_norm, gt_apex_norm)
        apex_attach = self._sample_maps(gt_mesial_dt,
                                        pred_apex[:, None]).mean() + self._sample_maps(
                                            gt_distal_dt,
                                            pred_apex[:, None]).mean()
        apex_loss = 0.5 * apex_reg + 0.25 * apex_attach

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
            'loss_recon_kpt':
            keypoint_loss * self.recon_kpt_weight,
            'loss_attach':
            attach_loss * self.attach_weight,
            'loss_order':
            order_loss * self.order_weight,
            'loss_apex':
            apex_loss * self.apex_weight,
        }

    def predict(self,
                feats,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}):
        del batch_data_samples
        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            outputs = self.forward(feats[0])
            outputs_flip = self.forward(feats[1])
            structure_logits, contours, keypoints = self._merge_tta(
                outputs, outputs_flip)
        else:
            structure_logits, contours, keypoints = self.forward(feats)

        structure_probs = structure_logits.sigmoid()
        mesial_scores = self._sample_maps(structure_probs[:, 1:2],
                                          keypoints[:, [0, 1]]).squeeze(-1)
        distal_scores = self._sample_maps(structure_probs[:, 2:3],
                                          keypoints[:, [3, 4]]).squeeze(-1)
        apex_score = 0.5 * (
            self._sample_maps(structure_probs[:, 1:2], keypoints[:, [2]]) +
            self._sample_maps(structure_probs[:, 2:3], keypoints[:, [2]])).squeeze(
                -1)
        keypoint_scores = torch.stack([
            mesial_scores[:, 0],
            mesial_scores[:, 1],
            apex_score[:, 0],
            distal_scores[:, 0],
            distal_scores[:, 1],
        ],
                                      dim=1).clamp(0.0, 1.0)

        pred_instances = []
        pred_fields = []
        for index in range(keypoints.shape[0]):
            keypoint_np = keypoints[index:index + 1].detach().cpu().numpy()
            keypoint_score_np = keypoint_scores[index:index + 1].detach().cpu().numpy()
            mesial_np = contours[index:index + 1, 0].detach().cpu().numpy()
            distal_np = contours[index:index + 1, 1].detach().cpu().numpy()
            pred_instances.append(
                InstanceData(
                    keypoints=keypoint_np,
                    keypoint_scores=keypoint_score_np,
                    mesial_contour=mesial_np,
                    distal_contour=distal_np))
            pred_fields.append(
                PixelData(
                    root_mask=structure_probs[index, 0].detach(),
                    mesial_boundary=structure_probs[index, 1].detach(),
                    distal_boundary=structure_probs[index, 2].detach()))
        return pred_instances, pred_fields
