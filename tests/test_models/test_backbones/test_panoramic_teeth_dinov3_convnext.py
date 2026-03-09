import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _patch_torch_pytree():
    import torch.utils._pytree as torch_pytree

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


_patch_torch_pytree()

pytest.importorskip('mmcv')
pytest.importorskip('mmengine')
pytest.importorskip('transformers')

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeHFModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 3, 1)) for _ in range(4)
        ])
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        batch_size, _, height, width = pixel_values.shape
        hidden_states = (
            torch.randn(batch_size, 96, height // 4, width // 4),
            torch.randn(batch_size, 96, height // 4, width // 4),
            torch.randn(batch_size, 192, height // 8, width // 8),
            torch.randn(batch_size, 384, height // 16, width // 16),
            torch.randn(batch_size, 768, height // 32, width // 32),
        )
        return SimpleNamespace(hidden_states=hidden_states)


def _patch_transformers(monkeypatch, recorder=None):
    import transformers

    def _config_from_pretrained(cls, pretrained, local_files_only=True):
        if recorder is not None:
            recorder.append(
                ('config', pretrained, dict(local_files_only=local_files_only)))
        return SimpleNamespace(hidden_sizes=[96, 192, 384, 768])

    def _model_from_pretrained(cls,
                               pretrained,
                               local_files_only=True,
                               trust_remote_code=False):
        if recorder is not None:
            recorder.append(
                ('model', pretrained,
                 dict(
                     local_files_only=local_files_only,
                     trust_remote_code=trust_remote_code)))
        return _FakeHFModel()

    monkeypatch.setattr(
        transformers.AutoConfig,
        'from_pretrained',
        classmethod(_config_from_pretrained))
    monkeypatch.setattr(
        transformers.AutoModel,
        'from_pretrained',
        classmethod(_model_from_pretrained))


def _make_fake_checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / 'dinov3-convnext-small-pretrain-lvd1689m'
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'config.json').write_text('{}', encoding='utf-8')
    (checkpoint_dir / 'model.safetensors').write_text('fake', encoding='utf-8')
    return checkpoint_dir


def test_dinov3_backbone_forward_and_freeze(monkeypatch, tmp_path):
    _patch_transformers(monkeypatch)
    checkpoint_dir = _make_fake_checkpoint_dir(tmp_path)

    from projects.panoramic_teeth.models import DINOv3ConvNextBackbone

    model = DINOv3ConvNextBackbone(
        pretrained=str(checkpoint_dir), out_indices=(3, ), trainable_stages=())

    outputs = model(torch.randn(1, 3, 384, 256))

    assert len(outputs) == 1
    assert outputs[0].shape == (1, 768, 12, 8)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_dinov3_backbone_unfreezes_only_last_stage(monkeypatch, tmp_path):
    _patch_transformers(monkeypatch)
    checkpoint_dir = _make_fake_checkpoint_dir(tmp_path)

    from projects.panoramic_teeth.models import DINOv3ConvNextBackbone

    model = DINOv3ConvNextBackbone(
        pretrained=str(checkpoint_dir),
        out_indices=(3, ),
        trainable_stages=(3, ))

    stage_flags = {
        name: parameter.requires_grad
        for name, parameter in model.model.named_parameters()
    }

    assert stage_flags['stages.3.0.weight']
    assert stage_flags['stages.3.0.bias']
    assert stage_flags['layer_norm.weight']
    assert stage_flags['layer_norm.bias']
    assert not stage_flags['stages.2.0.weight']
    assert not stage_flags['stages.2.0.bias']


def test_dinov3_backbone_requires_complete_local_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / 'dinov3-convnext-small-pretrain-lvd1689m'
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'config.json').write_text('{}', encoding='utf-8')

    from projects.panoramic_teeth.models import DINOv3ConvNextBackbone

    with pytest.raises(FileNotFoundError, match='model.safetensors'):
        DINOv3ConvNextBackbone(pretrained=str(checkpoint_dir))


def test_dinov3_backbone_uses_local_files_only(monkeypatch, tmp_path):
    recorder = []
    _patch_transformers(monkeypatch, recorder=recorder)
    checkpoint_dir = _make_fake_checkpoint_dir(tmp_path)

    from projects.panoramic_teeth.models import DINOv3ConvNextBackbone

    DINOv3ConvNextBackbone(pretrained=str(checkpoint_dir))

    assert recorder == [
        ('config', str(checkpoint_dir), dict(local_files_only=True)),
        (
            'model',
            str(checkpoint_dir),
            dict(local_files_only=True, trust_remote_code=False),
        ),
    ]
