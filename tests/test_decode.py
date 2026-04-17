import pytest

torch = pytest.importorskip("torch")

from cape_det.models.heads.decode import decode_anchor_free_outputs


def test_decode_anchor_free_outputs_shapes():
    outputs = {
        "p3": {
            "class_logits": torch.zeros(1, 1, 2, 2),
            "box_reg": torch.ones(1, 4, 2, 2),
            "objectness": torch.zeros(1, 1, 2, 2),
        }
    }
    preds = decode_anchor_free_outputs(outputs, [(32, 32)], score_threshold=0.1)
    assert preds[0]["boxes"].shape[1] == 4
    assert preds[0]["labels"].dtype == torch.long
