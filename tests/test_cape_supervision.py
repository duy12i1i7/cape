import pytest

torch = pytest.importorskip("torch")

from cape_det.losses.cape_losses import cape_hypothesis_loss, footprint_consistency_loss, refinement_improvement_targets
from cape_det.models.cape.cape_branch import CapeBranch
from cape_det.models.cape.types import CapeInternals, CapePredictions, HypothesisReadout


def _center_size(boxes_norm):
    centers = (boxes_norm[..., :2] + boxes_norm[..., 2:]) * 0.5
    sizes = (boxes_norm[..., 2:] - boxes_norm[..., :2]).clamp(min=0.0)
    return centers, sizes


def _readout(boxes, boxes_norm, scores=None, conf_logits=None):
    batch, count, _ = boxes.shape
    if conf_logits is None:
        conf_logits = torch.zeros(batch, count)
    if scores is None:
        scores = conf_logits.sigmoid()
    centers, sizes = _center_size(boxes_norm)
    return HypothesisReadout(
        boxes=boxes,
        scores=scores,
        labels=torch.zeros(batch, count, dtype=torch.long),
        class_logits=torch.zeros(batch, count, 1),
        conf_logits=conf_logits,
        conf_scores=conf_logits.sigmoid(),
        centers_norm=centers,
        sizes_norm=sizes,
        boxes_norm=boxes_norm,
    )


def _make_cape(seed_readout, final_readout, seed_compatibility=None, final_compatibility=None):
    batch, count, _ = final_readout.boxes.shape
    seed_value_logits = torch.zeros(batch, count)
    internals = CapeInternals(
        seed_params=torch.zeros(batch, count, 16),
        final_params=torch.zeros(batch, count, 16),
        latent=torch.zeros(batch, count, 64),
        seed_readout=seed_readout,
        final_readout=final_readout,
        seed_footprints=None,
        final_footprints=None,
        seed_evidence_maps=None,
        final_evidence_maps=None,
        seed_compatibility=seed_compatibility,
        final_compatibility=final_compatibility,
        seed_value_logits=seed_value_logits,
        final_value_logits=torch.zeros(batch, count),
        footprints=None,
        evidence_maps=None,
        compatibility=final_compatibility,
        value_logits=seed_value_logits,
        active_masks=None,
        avg_active_hypotheses=torch.tensor(0.0),
        avg_refinement_budget_used=torch.tensor(0.0),
    )
    return CapePredictions(
        boxes=final_readout.boxes,
        scores=final_readout.scores,
        labels=final_readout.labels,
        class_logits=final_readout.class_logits,
        internals=internals,
    )


def test_cape_branch_stores_seed_and_final_readouts():
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

    assert out.internals.seed_readout.boxes.shape == (1, 8, 4)
    assert out.internals.final_readout.boxes.shape == (1, 8, 4)
    assert out.internals.seed_readout.conf_logits.shape == (1, 8)
    assert out.internals.final_readout.boxes_norm.shape == (1, 8, 4)
    assert torch.equal(out.boxes, out.internals.final_readout.boxes)
    assert torch.equal(out.scores, out.internals.final_readout.scores)


def test_cape_box_center_size_losses_are_normalized_across_image_scale():
    target_small = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "labels": torch.tensor([0]),
        "ignore": torch.tensor([False]),
        "size": (100, 100),
    }
    target_large = {
        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
        "labels": torch.tensor([0]),
        "ignore": torch.tensor([False]),
        "size": (1000, 1000),
    }
    pred_norm = torch.tensor([[[0.11, 0.10, 0.21, 0.20]]])
    small_readout = _readout(torch.tensor([[[11.0, 10.0, 21.0, 20.0]]]), pred_norm, scores=torch.ones(1, 1))
    large_readout = _readout(torch.tensor([[[110.0, 100.0, 210.0, 200.0]]]), pred_norm, scores=torch.ones(1, 1))
    cape_small = _make_cape(small_readout, small_readout)
    cape_large = _make_cape(large_readout, large_readout)

    small_losses = cape_hypothesis_loss(cape_small, [target_small], {"footprint": 0.0, "sparsity": 0.0, "value": 0.0})
    large_losses = cape_hypothesis_loss(cape_large, [target_large], {"footprint": 0.0, "sparsity": 0.0, "value": 0.0})

    assert torch.allclose(small_losses["cape_box_loss"], large_losses["cape_box_loss"], atol=1e-6)
    assert torch.allclose(small_losses["cape_center_loss"], large_losses["cape_center_loss"], atol=1e-6)
    assert torch.allclose(small_losses["cape_size_loss"], large_losses["cape_size_loss"], atol=1e-6)


def test_cape_confidence_loss_uses_logits_and_stays_finite():
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]])
    boxes_norm = boxes / 100.0
    readout = _readout(boxes, boxes_norm, scores=torch.tensor([[1.0, 0.0]]), conf_logits=torch.tensor([[50.0, -50.0]]))
    cape = _make_cape(readout, readout)
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([0]),
        "ignore": torch.tensor([False]),
        "size": (100, 100),
    }

    losses = cape_hypothesis_loss(cape, [target], {"footprint": 0.0, "sparsity": 0.0, "value": 0.0})

    assert torch.isfinite(losses["cape_conf_loss"])


def test_footprint_loss_trains_footprint_not_evidence_target():
    footprints = torch.full((1, 2, 1, 3, 3), 0.2, requires_grad=True)
    evidence_maps = torch.full((1, 2, 1, 3, 3), 2.0, requires_grad=True)

    loss = footprint_consistency_loss(footprints, evidence_maps)
    loss.backward()

    assert footprints.grad is not None
    assert footprints.grad.abs().sum() > 0
    assert evidence_maps.grad is None


def test_value_targets_use_detached_refinement_improvement():
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([0]),
        "ignore": torch.tensor([False]),
        "size": (100, 100),
    }
    gt_norm = torch.tensor([[[0.0, 0.0, 0.1, 0.1]]])
    far_norm = torch.tensor([[[0.2, 0.2, 0.3, 0.3]]])
    seed_far = _readout(torch.tensor([[[20.0, 20.0, 30.0, 30.0]]]), far_norm, scores=torch.tensor([[0.1]]))
    final_good = _readout(torch.tensor([[[0.0, 0.0, 10.0, 10.0]]]), gt_norm, scores=torch.tensor([[0.9]]))
    improved = _make_cape(seed_far, final_good, torch.tensor([[[0.2]]]), torch.tensor([[[0.8]]]))

    value_targets = refinement_improvement_targets(improved, [target])

    assert value_targets.requires_grad is False
    assert value_targets[0, 0] > 0.7

    seed_good = _readout(torch.tensor([[[0.0, 0.0, 10.0, 10.0]]]), gt_norm, scores=torch.tensor([[0.9]]))
    final_far = _readout(torch.tensor([[[20.0, 20.0, 30.0, 30.0]]]), far_norm, scores=torch.tensor([[0.1]]))
    worsened = _make_cape(seed_good, final_far, torch.tensor([[[0.8]]]), torch.tensor([[[0.2]]]))

    value_targets = refinement_improvement_targets(worsened, [target])

    assert torch.equal(value_targets, torch.zeros_like(value_targets))
