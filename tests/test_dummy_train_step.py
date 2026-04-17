import pytest

torch = pytest.importorskip("torch")

from cape_det.losses.total_loss import CompositeDetectionLoss
from cape_det.models import build_model


def test_dummy_train_step_backward():
    config = {
        "model": {
            "mode": "baseline",
            "num_classes": 1,
            "fpn_channels": 128,
            "cape": {"enabled": False},
        }
    }
    model = build_model(config)
    criterion = CompositeDetectionLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn(1, 3, 96, 96)
    targets = [
        {
            "boxes": torch.tensor([[20.0, 20.0, 30.0, 34.0]]),
            "labels": torch.tensor([0]),
            "ignore": torch.tensor([False]),
            "size": (96, 96),
            "orig_size": (96, 96),
        }
    ]
    loss = criterion(model(images, targets), targets)["loss"]
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss.detach())
