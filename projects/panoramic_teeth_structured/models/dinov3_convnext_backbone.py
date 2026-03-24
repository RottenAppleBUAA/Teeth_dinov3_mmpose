from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones.base_backbone import BaseBackbone
from mmpose.registry import MODELS


@MODELS.register_module()
class DINOv3ConvNextBackbone(BaseBackbone):
    """Standalone local DINOv3 ConvNeXt backbone for structured teeth tasks."""

    _NUM_STAGES = 4

    def __init__(self,
                 pretrained: str,
                 out_indices: Sequence[int] = (3, ),
                 trainable_stages: Sequence[int] = (),
                 norm_eval: bool = True,
                 local_files_only: bool = True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.pretrained = str(Path(pretrained))
        self.out_indices = tuple(int(index) for index in out_indices)
        self.trainable_stages = tuple(int(index) for index in trainable_stages)
        self.norm_eval = bool(norm_eval)
        self.local_files_only = bool(local_files_only)

        self._validate_checkpoint_dir()
        self._validate_stage_indices()

        auto_config_cls, auto_model_cls = self._import_transformers()
        self.config = auto_config_cls.from_pretrained(
            self.pretrained, local_files_only=self.local_files_only)
        self.hidden_sizes = tuple(getattr(self.config, 'hidden_sizes', ()))
        if len(self.hidden_sizes) != self._NUM_STAGES:
            raise ValueError(
                'DINOv3 ConvNeXt backbone expects 4 hidden sizes, '
                f'but got {self.hidden_sizes!r}.')

        try:
            self.model = auto_model_cls.from_pretrained(
                self.pretrained,
                local_files_only=self.local_files_only,
                trust_remote_code=False)
        except Exception as exc:
            raise RuntimeError(
                'Failed to load the local DINOv3 ConvNeXt backbone. '
                'Ensure the installed `transformers` build provides '
                '`DINOv3ConvNextModel` and is compatible with the current '
                f'PyTorch version. Original error: {exc}') from exc

        self.out_channels = tuple(self.hidden_sizes[index]
                                  for index in self.out_indices)
        self._apply_trainable_stages()

    def _validate_checkpoint_dir(self) -> None:
        checkpoint_dir = Path(self.pretrained)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f'DINOv3 checkpoint directory does not exist: {checkpoint_dir}')

        required_files = ('config.json', 'model.safetensors')
        missing_files = [
            file_name for file_name in required_files
            if not (checkpoint_dir / file_name).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                'DINOv3 checkpoint directory is incomplete. Missing files: '
                f'{missing_files!r} in {checkpoint_dir}')

    def _validate_stage_indices(self) -> None:
        for name, indices in (('out_indices', self.out_indices),
                              ('trainable_stages', self.trainable_stages)):
            if not indices:
                if name == 'out_indices':
                    raise ValueError('out_indices must contain at least one index.')
                continue
            for index in indices:
                if index < 0 or index >= self._NUM_STAGES:
                    raise ValueError(
                        f'{name} must be in [0, {self._NUM_STAGES - 1}], '
                        f'but got {indices!r}.')

    @staticmethod
    def _import_transformers():
        DINOv3ConvNextBackbone._patch_torch_pytree_compat()
        try:
            from transformers import AutoConfig, AutoModel
        except Exception as exc:
            raise RuntimeError(
                'DINOv3ConvNextBackbone requires the `transformers` package. '
                'Install a DINOv3-compatible `transformers` build together '
                'with `safetensors`.') from exc
        return AutoConfig, AutoModel

    @staticmethod
    def _patch_torch_pytree_compat() -> None:
        try:
            import torch.utils._pytree as torch_pytree
        except Exception:
            return

        if hasattr(torch_pytree, 'register_pytree_node'):
            return
        if not hasattr(torch_pytree, '_register_pytree_node'):
            return

        def _compat_register_pytree_node(typ,
                                         flatten_fn,
                                         unflatten_fn,
                                         *,
                                         serialized_type_name=None,
                                         to_dumpable_context=None,
                                         from_dumpable_context=None):
            del serialized_type_name
            return torch_pytree._register_pytree_node(
                typ,
                flatten_fn,
                unflatten_fn,
                to_dumpable_context=to_dumpable_context,
                from_dumpable_context=from_dumpable_context)

        torch_pytree.register_pytree_node = _compat_register_pytree_node

    @staticmethod
    def _set_module_trainable(module: Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = True

    def _get_optional_submodule(self, path: str) -> Optional[Module]:
        module = self.model
        for attr in path.split('.'):
            if not hasattr(module, attr):
                return None
            module = getattr(module, attr)
        return module

    def _get_stage_module(self, index: int) -> Optional[Module]:
        for path in ('encoder.stages', 'stages'):
            stages = self._get_optional_submodule(path)
            if stages is None:
                continue
            try:
                return stages[index]
            except (IndexError, KeyError, TypeError):
                continue
        return None

    def _apply_trainable_stages(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if not self.trainable_stages:
            return

        missing_stages = []
        for index in self.trainable_stages:
            stage_module = self._get_stage_module(index)
            if stage_module is None:
                missing_stages.append(index)
                continue
            self._set_module_trainable(stage_module)

        if missing_stages:
            raise RuntimeError(
                'Unable to locate DINOv3 stage modules for '
                f'{missing_stages!r}.')

        if max(self.trainable_stages) == self._NUM_STAGES - 1:
            for path in (
                    'layernorm', 'post_layernorm', 'norm', 'layer_norm',
                    'encoder.layernorm', 'encoder.post_layernorm',
                    'encoder.norm', 'encoder.layer_norm'):
                module = self._get_optional_submodule(path)
                if module is not None:
                    self._set_module_trainable(module)

    def _get_stage_features(self, outputs) -> Tuple[Tensor, ...]:
        features = getattr(outputs, 'reshaped_hidden_states', None)
        if not features:
            features = getattr(outputs, 'hidden_states', None)
        if not features:
            raise RuntimeError(
                'DINOv3 backbone did not return hidden states. Ensure the '
                'underlying model supports output_hidden_states=True.')

        features = tuple(features)
        if len(features) == len(self.hidden_sizes) + 1:
            features = features[1:]
        elif len(features) > len(self.hidden_sizes):
            features = features[-len(self.hidden_sizes):]

        if len(features) != len(self.hidden_sizes):
            raise RuntimeError(
                'Unexpected number of DINOv3 hidden states: '
                f'expected {len(self.hidden_sizes)}, got {len(features)}.')

        stage_features = []
        for feature, expected_channels in zip(features, self.hidden_sizes):
            if feature.ndim != 4:
                raise RuntimeError(
                    'DINOv3 ConvNeXt hidden states must be 4D feature maps, '
                    f'but got shape {tuple(feature.shape)}.')
            if (feature.shape[1] != expected_channels
                    and feature.shape[-1] == expected_channels):
                feature = feature.permute(0, 3, 1, 2).contiguous()
            stage_features.append(feature)

        return tuple(stage_features[index] for index in self.out_indices)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outputs = self.model(
            pixel_values=x, output_hidden_states=True, return_dict=True)
        return self._get_stage_features(outputs)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()
        return self

