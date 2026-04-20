import pytest

torch = pytest.importorskip("torch")

from cape_det.losses.total_loss import CompositeDetectionLoss
from cape_det.losses.detection_losses import assign_anchor_free_targets
from cape_det.models import build_model


def test_composite_loss_runs_on_dummy_batch():
    config = {
        "model": {
            "mode": "cape",
            "num_classes": 1,
            "fpn_channels": 128,
            "cape": {
                "enabled": True,
                "num_hypotheses": 8,
                "max_active_hypotheses": 4,
                "max_refinement_steps": 1,
                "footprint_size": 7,
            },
        },
        "loss": {"sparsity": 0.01, "value": 0.01},
    }
    model = build_model(config)
    criterion = CompositeDetectionLoss(config)
    images = torch.randn(1, 3, 96, 96)
    targets = [
        {
            "boxes": torch.tensor([[20.0, 20.0, 28.0, 32.0]]),
            "labels": torch.tensor([0]),
            "ignore": torch.tensor([False]),
            "size": (96, 96),
            "orig_size": (96, 96),
        }
    ]
    losses = criterion(model(images, targets), targets)
    assert torch.isfinite(losses["loss"])


def test_anchor_free_targets_mask_ignored_regions():
    outputs = {
        "p3": {
            "class_logits": torch.zeros(1, 1, 8, 8),
            "box_reg": torch.zeros(1, 4, 8, 8),
            "objectness": torch.zeros(1, 1, 8, 8),
        }
    }
    targets = [
        {
            "boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]]),
            "labels": torch.tensor([-1]),
            "ignore": torch.tensor([True]),
            "size": (64, 64),
            "orig_size": (64, 64),
        }
    ]
    assigned = assign_anchor_free_targets(outputs, targets, num_classes=1, strides={"p3": 8})
    assert assigned["p3"]["positive"].sum().item() == 0
    assert assigned["p3"]["ignore"].sum().item() > 0
