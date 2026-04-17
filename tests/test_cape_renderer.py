import pytest

torch = pytest.importorskip("torch")

from cape_det.models.cape.primitive_vocabulary import PrimitiveVocabulary
from cape_det.models.cape.renderer import DifferentiableFootprintRenderer


def test_renderer_is_differentiable():
    vocab = PrimitiveVocabulary(num_primitives=4, footprint_size=7)
    renderer = DifferentiableFootprintRenderer(7)
    logits = torch.randn(2, 3, 4, requires_grad=True)
    blur = torch.randn(2, 3, requires_grad=True)
    out = renderer(vocab(logits), blur)
    assert out.shape == (2, 3, 1, 7, 7)
    out.mean().backward()
    assert logits.grad is not None
