import pytest

torch = pytest.importorskip("torch")

from cape_det.models.cape.cape_branch import CapeBranch


def test_cape_refinement_budget_shapes():
    branch = CapeBranch(
        128,
        1,
        {
            "num_hypotheses": 8,
            "max_active_hypotheses": 4,
            "max_refinement_steps": 2,
            "footprint_size": 7,
            "num_primitives": 4,
        },
    )
    features = {"p3": torch.randn(1, 128, 12, 12)}
    out = branch(features, [(96, 96)])
    assert out.boxes.shape == (1, 8, 4)
    assert out.internals.active_masks.shape == (1, 2, 8)
    assert out.internals.avg_refinement_budget_used.item() == 8.0
