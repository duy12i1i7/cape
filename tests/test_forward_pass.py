import pytest

torch = pytest.importorskip("torch")

from cape_det.models import build_model


def test_hybrid_forward_baseline_and_cape():
    config = {
        "model": {
            "mode": "cape",
            "num_classes": 1,
            "fpn_channels": 128,
            "score_threshold": 0.01,
            "max_detections": 20,
            "cape": {
                "enabled": True,
                "num_hypotheses": 8,
                "max_active_hypotheses": 4,
                "max_refinement_steps": 1,
                "num_primitives": 4,
                "footprint_size": 7,
            },
        }
    }
    model = build_model(config)
    model.eval()
    images = torch.randn(2, 3, 96, 96)
    targets = [{"size": (96, 96), "orig_size": (96, 96)}, {"size": (96, 96), "orig_size": (96, 96)}]
    out = model(images, targets)
    assert len(out["predictions"]) == 2
    assert out["predictions_decoded"] is True
    assert out["cape"].boxes.shape[:2] == (2, 8)


def test_hybrid_forward_skips_decode_during_train_by_default():
    config = {
        "model": {
            "mode": "cape",
            "num_classes": 1,
            "fpn_channels": 128,
            "score_threshold": 0.01,
            "max_detections": 20,
            "cape": {
                "enabled": True,
                "num_hypotheses": 8,
                "max_active_hypotheses": 4,
                "max_refinement_steps": 1,
                "num_primitives": 4,
                "footprint_size": 7,
            },
        }
    }
    model = build_model(config)
    model.train()
    images = torch.randn(2, 3, 96, 96)
    targets = [{"size": (96, 96), "orig_size": (96, 96)}, {"size": (96, 96), "orig_size": (96, 96)}]
    out = model(images, targets)

    assert out["predictions_decoded"] is False
    assert len(out["predictions"]) == 2
    assert all(pred["boxes"].numel() == 0 for pred in out["predictions"])
    assert out["global_raw"]
    assert out["cape"].boxes.shape[:2] == (2, 8)


def test_hybrid_forward_can_decode_during_train_when_requested():
    config = {
        "model": {
            "mode": "baseline",
            "num_classes": 1,
            "fpn_channels": 128,
            "score_threshold": 0.0,
            "max_detections": 20,
            "decode_during_train": True,
            "cape": {"enabled": False},
        }
    }
    model = build_model(config)
    model.train()
    images = torch.randn(1, 3, 96, 96)
    out = model(images, [{"size": (96, 96), "orig_size": (96, 96)}])

    assert out["predictions_decoded"] is True
    assert len(out["predictions"]) == 1
    assert out["predictions"][0]["boxes"].shape[1] == 4
