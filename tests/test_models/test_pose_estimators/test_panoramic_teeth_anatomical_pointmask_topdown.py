import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmengine.structures import InstanceData, PixelData
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

from mmengine.config import Config  # noqa: E402
from mmengine.runner.checkpoint import load_checkpoint  # noqa: E402

from mmpose.registry import KEYPOINT_CODECS  # noqa: E402
from mmpose.structures import PoseDataSample  # noqa: E402
from mmpose.testing import get_packed_inputs  # noqa: E402
from mmpose.utils import register_all_modules  # noqa: E402


class _FakeHFModel(nn.Module):

    def __init__(self, hidden_sizes=(96, 192, 384, 768)):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 3, 1)) for _ in range(4)
        ])
        self.layer_norm = nn.LayerNorm(self.hidden_sizes[-1])

    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        del output_hidden_states, return_dict
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device
        hidden_states = (
            torch.randn(
                batch_size,
                self.hidden_sizes[0],
                height // 4,
                width // 4,
                device=device),
            torch.randn(
                batch_size,
                self.hidden_sizes[0],
                height // 4,
                width // 4,
                device=device),
            torch.randn(
                batch_size,
                self.hidden_sizes[1],
                height // 8,
                width // 8,
                device=device),
            torch.randn(
                batch_size,
                self.hidden_sizes[2],
                height // 16,
                width // 16,
                device=device),
            torch.randn(
                batch_size,
                self.hidden_sizes[3],
                height // 32,
                width // 32,
                device=device),
        )
        return SimpleNamespace(hidden_states=hidden_states)


def _patch_transformers(monkeypatch, hidden_sizes=(96, 192, 384, 768)):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        'from_pretrained',
        classmethod(lambda cls, pretrained, local_files_only=True: SimpleNamespace(hidden_sizes=list(hidden_sizes))))
    monkeypatch.setattr(
        transformers.AutoModel,
        'from_pretrained',
        classmethod(lambda cls, pretrained, local_files_only=True, trust_remote_code=False: _FakeHFModel(hidden_sizes=hidden_sizes)))


def _make_fake_checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / 'dinov3-convnext-small-pretrain-lvd1689m'
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'config.json').write_text('{}', encoding='utf-8')
    (checkpoint_dir / 'model.safetensors').write_text('fake', encoding='utf-8')
    return checkpoint_dir


def _make_targets():
    height, width = 512, 192
    keypoints = torch.tensor(
        [[44.0, 96.0], [68.0, 168.0], [96.0, 440.0], [124.0, 168.0],
         [148.0, 96.0]],
        dtype=torch.float32)

    root_mask = torch.zeros(1, height, width, dtype=torch.float32)
    root_mask[:, 96:452, 44:148] = 1.0

    mesial_polyline_map = torch.zeros(1, height, width, dtype=torch.float32)
    mesial_polyline_map[:, 96:441, 44:69] = 1.0
    distal_polyline_map = torch.zeros(1, height, width, dtype=torch.float32)
    distal_polyline_map[:, 96:441, 124:149] = 1.0

    mesial_polyline_distance = torch.zeros(1, height, width, dtype=torch.float32)
    distal_polyline_distance = torch.zeros(1, height, width, dtype=torch.float32)

    return dict(
        keypoints=keypoints,
        root_mask=root_mask,
        mesial_polyline_map=mesial_polyline_map,
        distal_polyline_map=distal_polyline_map,
        mesial_polyline_distance=mesial_polyline_distance,
        distal_polyline_distance=distal_polyline_distance)


def _make_packed_inputs(batch_size=2):
    packed_inputs = get_packed_inputs(
        batch_size=batch_size,
        num_instances=1,
        num_keypoints=5,
        img_shape=(512, 192),
        input_size=(192, 512),
        with_heatmap=False,
        with_reg_label=False,
        with_simcc_label=False)
    targets = _make_targets()
    codec = KEYPOINT_CODECS.build(
        dict(
            type='SimCCLabel',
            input_size=(192, 512),
            sigma=(5.66, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False))

    mesial_encoded = codec.encode(
        keypoints=targets['keypoints'][None, [0, 1]].numpy(),
        keypoints_visible=torch.ones(1, 2).numpy())
    apex_encoded = codec.encode(
        keypoints=targets['keypoints'][None, [2]].numpy(),
        keypoints_visible=torch.ones(1, 1).numpy())
    distal_encoded = codec.encode(
        keypoints=targets['keypoints'][None, [3, 4]].numpy(),
        keypoints_visible=torch.ones(1, 2).numpy())

    for data_sample in packed_inputs['data_samples']:
        data_sample.gt_fields = PixelData(
            root_mask=targets['root_mask'].clone(),
            mesial_polyline_map=targets['mesial_polyline_map'].clone(),
            distal_polyline_map=targets['distal_polyline_map'].clone(),
            mesial_polyline_distance=targets['mesial_polyline_distance'].clone(),
            distal_polyline_distance=targets['distal_polyline_distance'].clone())
        data_sample.gt_instance_labels = InstanceData(
            keypoint_targets=targets['keypoints'].clone().unsqueeze(0),
            keypoint_weights=torch.ones(1, 5, dtype=torch.float32),
            apex_midpoint_target=targets['keypoints'][2].clone().unsqueeze(0),
            mesial_keypoint_x_labels=torch.from_numpy(
                mesial_encoded['keypoint_x_labels']).float(),
            mesial_keypoint_y_labels=torch.from_numpy(
                mesial_encoded['keypoint_y_labels']).float(),
            mesial_keypoint_weights=torch.from_numpy(
                mesial_encoded['keypoint_weights']).float(),
            apex_keypoint_x_labels=torch.from_numpy(
                apex_encoded['keypoint_x_labels']).float(),
            apex_keypoint_y_labels=torch.from_numpy(
                apex_encoded['keypoint_y_labels']).float(),
            apex_keypoint_weights=torch.from_numpy(
                apex_encoded['keypoint_weights']).float(),
            distal_keypoint_x_labels=torch.from_numpy(
                distal_encoded['keypoint_x_labels']).float(),
            distal_keypoint_y_labels=torch.from_numpy(
                distal_encoded['keypoint_y_labels']).float(),
            distal_keypoint_weights=torch.from_numpy(
                distal_encoded['keypoint_weights']).float())
        data_sample.set_metainfo(
            dict(
                flip_indices=[4, 3, 2, 1, 0],
                dataset_name='panoramic_teeth_structured'))
    return packed_inputs


def _run_single_train_step(model, packed_inputs):
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3)
    optimizer.zero_grad()
    data = model.data_preprocessor(packed_inputs, training=True)
    losses = model.forward(**data, mode='loss')
    total_loss = sum(value for key, value in losses.items() if 'acc' not in key)
    total_loss.backward()
    optimizer.step()
    return losses


def test_base_anatomical_pointmask_config_builds_and_runs():
    register_all_modules()
    importlib.import_module('projects.panoramic_teeth_structured')

    config_path = ROOT / 'projects' / 'panoramic_teeth_structured' / 'configs' / (
        'panoramic-teeth-anatomical-pointmask_r50_8xb32-200e_v2-192x512.py')
    cfg = Config.fromfile(str(config_path))
    cfg.model.test_cfg.flip_test = False

    from mmpose.models import build_pose_estimator

    model = build_pose_estimator(cfg.model)
    packed_inputs = _make_packed_inputs(batch_size=2)
    data = model.data_preprocessor(packed_inputs, training=True)

    losses = model.forward(**data, mode='loss')
    assert 'loss_root_bce' in losses
    assert 'loss_kpt_mesial' in losses
    assert 'loss_kpt_apex' in losses
    assert 'loss_kpt_distal' in losses
    assert 'loss_side_attach' in losses
    assert 'loss_apex_consistency' in losses
    assert 'loss_contour' not in losses

    model.eval()
    with torch.no_grad():
        batch_results = model.forward(**data, mode='predict')

    assert len(batch_results) == 2
    assert isinstance(batch_results[0], PoseDataSample)
    assert hasattr(batch_results[0].pred_instances, 'keypoints')
    assert not hasattr(batch_results[0].pred_instances, 'mesial_contour')
    assert hasattr(batch_results[0].pred_fields, 'root_mask')


def test_stage1_and_stage2_anatomical_pointmask_dinov3_train_smoke(
        monkeypatch, tmp_path):
    _patch_transformers(monkeypatch)
    checkpoint_dir = _make_fake_checkpoint_dir(tmp_path)

    register_all_modules()
    importlib.import_module('projects.panoramic_teeth_structured')

    config_dir = ROOT / 'projects' / 'panoramic_teeth_structured' / 'configs'
    stage1_config_path = config_dir / (
        'panoramic-teeth-anatomical-pointmask_dinov3-convnext-s_8xb32-200e_'
        'v2-192x512_stage1.py')
    stage2_config_path = config_dir / (
        'panoramic-teeth-anatomical-pointmask_dinov3-convnext-s_8xb32-50e_'
        'v2-192x512_stage2.py')

    from mmpose.models import build_pose_estimator

    stage1_cfg = Config.fromfile(str(stage1_config_path))
    stage1_cfg.model.backbone.pretrained = str(checkpoint_dir)
    stage1_cfg.model.test_cfg.flip_test = False
    stage1_model = build_pose_estimator(stage1_cfg.model)

    packed_inputs = _make_packed_inputs(batch_size=2)
    stage1_losses = _run_single_train_step(stage1_model, packed_inputs)
    assert 'loss_kpt_mesial' in stage1_losses
    assert 'loss_root_bce' in stage1_losses

    stage1_dir = (
        tmp_path / 'work_dirs' /
        'panoramic-teeth-anatomical-pointmask_dinov3-convnext-s_8xb32-200e_v2-192x512_stage1'
    )
    stage1_dir.mkdir(parents=True)
    stage1_ckpt = stage1_dir / 'best_NME_epoch_1.pth'
    torch.save({'state_dict': stage1_model.state_dict()}, stage1_ckpt)

    monkeypatch.chdir(tmp_path)

    stage2_cfg = Config.fromfile(str(stage2_config_path))
    stage2_cfg.model.backbone.pretrained = str(checkpoint_dir)
    stage2_cfg.model.test_cfg.flip_test = False
    assert stage2_cfg.load_from is not None

    stage2_model = build_pose_estimator(stage2_cfg.model)
    load_checkpoint(stage2_model, stage2_cfg.load_from, map_location='cpu')

    stage2_losses = _run_single_train_step(stage2_model, packed_inputs)
    assert 'loss_side_attach' in stage2_losses
    assert 'loss_apex_consistency' in stage2_losses


def test_stage1_and_stage2_anatomical_pointmask_dinov3_base_configs_build(
        monkeypatch, tmp_path):
    _patch_transformers(monkeypatch, hidden_sizes=(128, 256, 512, 1024))
    checkpoint_dir = _make_fake_checkpoint_dir(tmp_path)

    register_all_modules()
    importlib.import_module('projects.panoramic_teeth_structured')

    config_dir = ROOT / 'projects' / 'panoramic_teeth_structured' / 'configs'
    stage1_config_path = config_dir / (
        'panoramic-teeth-anatomical-pointmask_dinov3-convnext-b_8xb32-200e_'
        'v2-192x512_stage1.py')
    stage2_config_path = config_dir / (
        'panoramic-teeth-anatomical-pointmask_dinov3-convnext-b_8xb32-50e_'
        'v2-192x512_stage2.py')

    from mmpose.models import build_pose_estimator

    stage1_cfg = Config.fromfile(str(stage1_config_path))
    stage1_cfg.model.backbone.pretrained = str(checkpoint_dir)
    stage1_model = build_pose_estimator(stage1_cfg.model)
    assert stage1_model.backbone.out_channels == (1024, )

    stage1_dir = (
        tmp_path / 'work_dirs' /
        'panoramic-teeth-anatomical-pointmask_dinov3-convnext-b_8xb32-200e_v2-192x512_stage1'
    )
    stage1_dir.mkdir(parents=True)
    torch.save({'state_dict': stage1_model.state_dict()},
               stage1_dir / 'best_NME_epoch_1.pth')

    monkeypatch.chdir(tmp_path)
    stage2_cfg = Config.fromfile(str(stage2_config_path))
    assert stage2_cfg.load_from is not None
