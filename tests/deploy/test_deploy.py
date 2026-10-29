"""Export and fast segmenter tests (M4-T01).

A trained-model folder is built from seg-model v1's own plans (models/registry/seg-model/v1/nnunet) with random
weights; the network is shrunk so the tests run on a CPU. The exported model must give the same mask as nnU-Net's own
predictor (what `nnUNetv2_predict` runs), with and without the 8 mirror flips.
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnunetv2")

from bmts.common.constants import MODALITIES  # noqa: E402
from bmts.deploy.export import MANIFEST_FILE, export_model, sha256  # noqa: E402
from bmts.deploy.segmenter import FastSegmenter  # noqa: E402
from bmts.uncertainty import nnunet_tta as T  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "models/registry/seg-model/v1/nnunet"


def small_plans():
    """seg-model v1's plans with a smaller network and patch (same layout: 6 stages, 3D, instance norm)."""
    plans = json.loads((V1 / "plans.json").read_text())
    cfg = plans["configurations"]["3d_fullres"]
    cfg["patch_size"] = [64, 64, 64]
    cfg["architecture"]["arch_kwargs"]["features_per_stage"] = [4, 8, 8, 8, 8, 8]
    return plans


def build_model_folder(root, plans, mirror_axes=(0, 1, 2), seed=0):
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    dataset_json = json.loads((V1 / "dataset.json").read_text())
    pm = PlansManager(copy.deepcopy(plans))
    torch.manual_seed(seed)
    net = nnUNetTrainer.build_network_architecture(pm, pm.get_configuration("3d_fullres"), 4, 4,
                                                   enable_deep_supervision=False)
    (root / "fold_0").mkdir(parents=True)
    (root / "plans.json").write_text(json.dumps(plans))
    (root / "dataset.json").write_text(json.dumps(dataset_json))
    torch.save({"network_weights": net.state_dict(), "trainer_name": "nnUNetTrainer",
                "init_args": {"configuration": "3d_fullres"}, "inference_allowed_mirroring_axes": mirror_axes},
               root / "fold_0" / "checkpoint_best.pth")
    return root


def write_scan(folder, shape=(70, 60, 50), seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    brain = np.zeros(shape, bool)
    brain[8:-8, 8:-8, 6:-6] = True
    for i, m in enumerate(MODALITIES):
        img = np.where(brain, rng.normal(1 + i, 0.5, shape), 0).astype(np.float32)
        img[30:40, 25:35, 20:30] += 3                                     # a bright blob
        nib.save(nib.Nifti1Image(img, np.diag([1.0, 1.0, 1.0, 1.0])), folder / f"{m}.nii.gz")
    return [folder / f"{m}.nii.gz" for m in MODALITIES]


def reference_mask(model_dir, files, tta):
    """What nnUNetv2_predict does: nnU-Net's predictor loaded from the checkpoint."""
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

    predictor = T.load_predictor(model_dir, device="cpu")
    predictor.use_mirroring = tta
    data, props = T.preprocess(predictor, files)
    logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))
    seg = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, predictor.plans_manager, predictor.configuration_manager, predictor.label_manager, props)
    return np.asarray(seg), logits


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("deploy")
    model_dir = build_model_folder(root / "trained", small_plans())
    manifest = export_model(model_dir, root / "export")
    files = write_scan(root / "scan")
    return model_dir, root / "export", manifest, files


def test_export_folder_is_complete(exported):
    model_dir, export_dir, manifest, _ = exported
    assert {p.name for p in export_dir.iterdir()} == {"seg_model.torchscript.pt", "plans.json", "dataset.json",
                                                      MANIFEST_FILE}
    assert manifest["source"]["checkpoint_sha256"] == sha256(model_dir / "fold_0" / "checkpoint_best.pth")
    assert manifest["model_sha256"] == sha256(export_dir / "seg_model.torchscript.pt")
    assert manifest["mirror_axes"] == [0, 1, 2] and manifest["patch_size"] == [64, 64, 64]
    assert manifest["export_check_max_abs_diff"] < 1e-3
    assert json.loads((export_dir / MANIFEST_FILE).read_text()) == manifest


@pytest.mark.parametrize("tta", [True, False])
def test_same_mask_as_nnunet(exported, tta):
    model_dir, export_dir, _, files = exported
    expected, expected_logits = reference_mask(model_dir, files, tta)
    segmenter = FastSegmenter.from_dir(export_dir, device="cpu", tta=tta)
    data, _ = T.preprocess(segmenter.predictor, files)
    assert torch.allclose(segmenter.logits(data), expected_logits, atol=1e-4)
    seg, props, timings = segmenter.segment(files)
    assert seg.shape == expected.shape == (50, 60, 70)                  # SimpleITK order (z, y, x) of the input
    assert np.array_equal(seg, expected)
    assert set(np.unique(seg)) <= {0, 1, 2, 3} and timings["total_s"] > 0


def test_segment_to_file_keeps_the_scan_geometry(exported, tmp_path):
    _, export_dir, _, files = exported
    seg, _ = FastSegmenter.from_dir(export_dir, device="cpu").segment_to_file(files, tmp_path / "out" / "seg.nii.gz")
    saved, scan = nib.load(tmp_path / "out" / "seg.nii.gz"), nib.load(files[0])
    assert saved.shape == scan.shape and np.allclose(saved.affine, scan.affine)
    assert np.array_equal(np.asarray(saved.dataobj).transpose(2, 1, 0), seg)


def test_wrong_checkpoint_or_changed_model_file_is_refused(exported, tmp_path):
    model_dir, export_dir, _, _ = exported
    with pytest.raises(ValueError, match="SHA-256"):
        export_model(model_dir, tmp_path / "x", expected_sha256="0" * 64)
    assert not (tmp_path / "x").exists()
    copy_dir = tmp_path / "export_copy"
    copy_dir.mkdir()
    for f in export_dir.iterdir():
        (copy_dir / f.name).write_bytes(f.read_bytes())
    with open(copy_dir / "seg_model.torchscript.pt", "ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="SHA-256"):
        FastSegmenter.from_dir(copy_dir, device="cpu")


def test_tta_needs_mirror_axes(tmp_path):
    model_dir = build_model_folder(tmp_path / "trained", small_plans(), mirror_axes=None)
    export_model(model_dir, tmp_path / "export")
    with pytest.raises(ValueError, match="mirror"):
        FastSegmenter.from_dir(tmp_path / "export", device="cpu", tta=True)


def test_full_size_v1_network_traces_exactly(tmp_path):
    """The real seg-model v1 architecture (32..320 features), on a smaller patch to keep the test quick."""
    plans = json.loads((V1 / "plans.json").read_text())
    plans["configurations"]["3d_fullres"]["patch_size"] = [64, 64, 64]
    manifest = export_model(build_model_folder(tmp_path / "trained", plans), tmp_path / "export")
    assert manifest["export_check_max_abs_diff"] < 1e-4 and manifest["output_channels"] == 4


def test_command_line(exported, tmp_path, monkeypatch):
    _, export_dir, _, files = exported
    path = REPO / "scripts" / "segment.py"
    spec = importlib.util.spec_from_file_location("segment_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["segment.py", "--export-dir", str(export_dir), "--scan", str(files[0].parent),
                                      "--device", "cpu", "--out", str(tmp_path / "out")])
    module.main()
    assert (tmp_path / "out" / "scan_seg.nii.gz").is_file() and (tmp_path / "out" / "timings.csv").is_file()
