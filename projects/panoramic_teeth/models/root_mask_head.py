from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.models.heads.base_head import BaseHead
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList


@MODELS.register_module()
class RootMaskHead(BaseHead):
    """A lightweight decoder that predicts a single-channel root mask."""

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 input_size: Tuple[int, int],
                 feat_channels: int = 256,
                 num_convs: int = 2,
                 bce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support multiple inputs.')

        self.input_size = input_size
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

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
        self.classifier = nn.Conv2d(current_channels, 1, kernel_size=1)

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        x = feats[-1]
        x = self.decoder(x)
        x = self.classifier(x)
        output_size = (int(self.input_size[1]), int(self.input_size[0]))
        return F.interpolate(
            x, size=output_size, mode='bilinear', align_corners=False)

    def _stack_targets(self, batch_data_samples: OptSampleList) -> Tensor:
        target = torch.stack(
            [sample.gt_fields.root_mask for sample in batch_data_samples],
            dim=0).float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        return target

    @staticmethod
    def _dice_loss(logits: Tensor, targets: Tensor,
                   eps: float = 1e-6) -> Tensor:
        probs = logits.sigmoid()
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + eps) / (denominator + eps)
        return 1 - dice.mean()

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        logits = self.forward(feats)
        targets = self._stack_targets(batch_data_samples).to(logits.device)

        loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
        loss_dice = self._dice_loss(logits, targets)

        return {
            'loss_mask_bce': loss_bce * self.bce_weight,
            'loss_mask_dice': loss_dice * self.dice_weight,
        }

    def predict(self,
                feats,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}):
        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            logits = self.forward(feats[0])
            logits_flip = self.forward(feats[1]).flip(-1)
            logits = 0.5 * (logits + logits_flip)
        else:
            logits = self.forward(feats)

        probs = logits.sigmoid()
        return [PixelData(root_mask=mask.detach()) for mask in probs]
