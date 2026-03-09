from __future__ import annotations

from itertools import zip_longest
from typing import Optional

from mmengine.structures import PixelData
from torch import Tensor

from mmpose.models.pose_estimators import TopdownPoseEstimator
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptConfigType, OptMultiConfig,
                                 SampleList)


@MODELS.register_module()
class TopdownRootMaskEstimator(TopdownPoseEstimator):
    """Topdown estimator with an auxiliary root-mask decoder."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        self.mask_head = MODELS.build(mask_head) if mask_head else None
        if self.mask_head is not None:
            self.mask_head.test_cfg = self.test_cfg.copy()

    @property
    def with_mask_head(self) -> bool:
        return self.mask_head is not None

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        feats = self.extract_feat(inputs)
        losses = {}

        if self.with_head:
            losses.update(
                self.head.loss(feats, data_samples, train_cfg=self.train_cfg))
        if self.with_mask_head:
            losses.update(
                self.mask_head.loss(
                    feats, data_samples, train_cfg=self.train_cfg))
        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        assert self.with_head, 'The model must have a keypoint head.'

        if self.test_cfg.get('flip_test', False):
            feats = [self.extract_feat(inputs), self.extract_feat(inputs.flip(-1))]
        else:
            feats = self.extract_feat(inputs)

        kp_preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)
        if isinstance(kp_preds, tuple):
            batch_pred_instances, batch_pred_fields = kp_preds
        else:
            batch_pred_instances = kp_preds
            batch_pred_fields = []

        if self.with_mask_head:
            mask_fields = self.mask_head.predict(
                feats, data_samples, test_cfg=self.test_cfg)
            if not batch_pred_fields:
                batch_pred_fields = [PixelData() for _ in batch_pred_instances]
            for pred_field, mask_field in zip_longest(batch_pred_fields,
                                                      mask_fields):
                if pred_field is None:
                    pred_field = PixelData()
                if mask_field is not None:
                    for key, value in mask_field.all_items():
                        pred_field.set_field(value, key)
            batch_pred_fields = batch_pred_fields[:len(batch_pred_instances)]

        return self.add_pred_to_datasample(batch_pred_instances,
                                           batch_pred_fields, data_samples)
