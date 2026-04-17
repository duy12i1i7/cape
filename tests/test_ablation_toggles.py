import pytest

torch = pytest.importorskip("torch")

from cape_det.losses.total_loss import CompositeDetectionLoss
from cape_det.models.cape.budget import HypothesisValueHead
from cape_det.models.cape.cape_branch import CapeBranch


def _features(batch=1):
    return {"p3": torch.randn(batch, 128, 12, 12)}


def test_renderer_toggle_changes_footprint_outputs():
    base = {
        "num_hypotheses": 8,
        "max_active_hypotheses": 4,
        "max_refinement_steps": 0,
        "enable_refinement": False,
        "footprint_size": 7,
        "num_primitives": 4,
    }
    with_renderer = CapeBranch(128, 1, {**base, "enable_renderer": True})
    no_renderer = CapeBranch(128, 1, {**base, "enable_renderer": False})

    rendered = with_renderer(_features(), [(96, 96)])
    skipped = no_renderer(_features(), [(96, 96)])

    assert rendered.internals.footprints is not None
    assert rendered.internals.evidence_maps is not None
    assert skipped.internals.footprints is None
    assert skipped.internals.evidence_maps is not None


def test_refinement_toggle_changes_budget_accounting_and_active_masks():
    base = {
        "num_hypotheses": 8,
        "max_active_hypotheses": 4,
        "footprint_size": 7,
        "num_primitives": 4,
    }
    refine = CapeBranch(128, 1, {**base, "enable_refinement": True, "max_refinement_steps": 2})
    no_refine = CapeBranch(128, 1, {**base, "enable_refinement": False, "max_refinement_steps": 2})

    refined = refine(_features(), [(96, 96)])
    frozen = no_refine(_features(), [(96, 96)])

    assert refined.internals.active_masks.shape == (1, 2, 8)
    assert refined.internals.avg_refinement_budget_used.item() == 8.0
    assert frozen.internals.active_masks is None
    assert frozen.internals.avg_refinement_budget_used.item() == 0.0


def test_value_head_toggle_changes_value_logits_source():
    params = torch.randn(1, 4, 16)
    latent = torch.randn(1, 4, 64)
    compatibility = torch.randn(1, 4, 1)
    disabled = HypothesisValueHead(16, 64, enabled=False)
    enabled = HypothesisValueHead(16, 64, enabled=True)

    disabled_logits = disabled(params, latent, compatibility)
    enabled_logits = enabled(params, latent, compatibility)

    assert torch.allclose(disabled_logits, params[..., 11])
    assert not torch.allclose(enabled_logits, params[..., 11])


def test_sparsity_and_value_calibration_toggles_change_loss_weights():
    config = {
        "model": {
            "num_classes": 1,
            "cape": {
                "enable_sparsity_loss": False,
                "enable_value_calibration": False,
            },
        },
        "loss": {
            "sparsity": 0.05,
            "value": 0.1,
        },
    }
    disabled = CompositeDetectionLoss(config)

    config_enabled = {
        "model": {
            "num_classes": 1,
            "cape": {
                "enable_sparsity_loss": True,
                "enable_value_calibration": True,
            },
        },
        "loss": {
            "sparsity": 0.05,
            "value": 0.1,
        },
    }
    enabled = CompositeDetectionLoss(config_enabled)

    assert disabled.loss_cfg["sparsity"] == 0.0
    assert disabled.loss_cfg["value"] == 0.0
    assert enabled.loss_cfg["sparsity"] == 0.05
    assert enabled.loss_cfg["value"] == 0.1
