import importlib
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

from mmengine.config import Config
from mmengine.structures import PixelData

from mmpose.structures import PoseDataSample
from mmpose.testing import get_packed_inputs
from mmpose.utils import register_all_modules


class _FakeHFModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 3, 1)) for _ in range(4)
        ])
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device
        hidden_states = (
            torch.randn(batch_size, 96, height // 4, width // 4, device=device),
            torch.randn(batch_size, 96, height // 4, width // 4, device=device),
            torch.randn(batch_size, 192, height // 8, width // 8, device=device),
            torch.randn(batch_size, 384, height // 16, width // 16, device=device),
            torch.randn(batch_size, 768, height // 32, width // 32, device=device),
        )
        return SimpleNamespace(hidden_states=hidden_states)


def _patch_transformers(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        'from_pretrained',
        classmethod(lambda cls, pretrained, local_files_only=True: SimpleNamespace(hidden_sizes=[96, 192, 384, 768])))
    monkeypatch.setattr(
        transformers.AutoModel,
        'from_pretrained',
        classmethod(lambda cls, pretrained, local_files_only=True, trust_remote_code=False: _FakeHFModel()))


def _make_fake_checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / 'dinov3-convnext-small-pretrain-lvd1689m'
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'config.json').write_text('{}', encoding='utf-8')
    (checkpoint_dir / 'model.safetensors').write_text('fake', encoding='utf-8')
    return checkpoint_dir


def _make_packed_inputs(batch_size=2):
    packed_inputs = get_packed_inputs(
        batch_size=batch_size,
        num_instances=1,
        num_keypoints=5,
        img_shape=(384, 256),
        input_size=(256, 384),
        with_heatmap=False,
        with_reg_label=False,
        with_simcc_label=True)
    for sample in packed_inputs:
        sample['data_samples'].gt_fields = PixelData(
            root_mask=torch.ones(1, 384, 256, dtype=torch.float32))
        sample['data_samples'].set_metainfo(
            dict(flip_indices=[0, 1, 2, 3, 4], dataset_name='panoramic_teeth'))
    return packed_inputs


def test_stage1_dinov3_config_builds_and_runs(monkeypatch, tmp_path):
    _patch_transformers(monkeypatch)
    checkpoint_dir = _make_fake_checkpoint_dir(tmp_path)

    register_all_modules()
    importlib.import_module('projects.panoramic_teeth')

    config_path = ROOT / 'projects' / 'panoramic_teeth' / 'configs' / (
        'rtmpose-dinov3-convnext-s_8xb32-60e_'
        'panoramic-teeth-v2-256x384_stage1.py')
    cfg = Config.fromfile(str(config_path))
    cfg.model.backbone.pretrained = str(checkpoint_dir)
    cfg.model.test_cfg.flip_test = False

    from mmpose.models import build_pose_estimator

    model = build_pose_estimator(cfg.model)

    packed_inputs = _make_packed_inputs(batch_size=2)
    data = model.data_preprocessor(packed_inputs, training=True)

    losses = model.forward(**data, mode='loss')
    assert 'loss_kpt' in losses
    assert 'loss_mask_bce' in losses
    assert 'loss_mask_dice' in losses

    model.eval()
    with torch.no_grad():
        batch_results = model.forward(**data, mode='predict')

    assert len(batch_results) == 2
    assert isinstance(batch_results[0], PoseDataSample)
    assert hasattr(batch_results[0].pred_instances, 'keypoints')
    assert hasattr(batch_results[0].pred_fields, 'root_mask')


def test_stage2_dinov3_config_resolves_stage1_best_checkpoint(monkeypatch,
                                                              tmp_path):
    stage1_dir = (
        tmp_path / 'work_dirs' /
        'rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1')
    stage1_dir.mkdir(parents=True)
    (stage1_dir / 'best_NME_epoch_5.pth').write_text('a', encoding='utf-8')
    (stage1_dir / 'best_NME_epoch_17.pth').write_text('b', encoding='utf-8')

    monkeypatch.chdir(tmp_path)

    config_path = ROOT / 'projects' / 'panoramic_teeth' / 'configs' / (
        'rtmpose-dinov3-convnext-s_8xb32-20e_'
        'panoramic-teeth-v2-256x384_stage2.py')
    cfg = Config.fromfile(str(config_path))

    assert cfg.load_from.replace('\\', '/').endswith('best_NME_epoch_17.pth')
    assert cfg.model.backbone.trainable_stages == (3, )
