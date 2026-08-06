import torch
import torch.nn.functional as F

from phaseek_v2.config import ModelConfig
from phaseek_v2.model import PhaseekV3Classifier


def test_graph_mixture_receives_classification_gradient():
    torch.manual_seed(7)
    config = ModelConfig(
        block_size=12,
        n_layer=2,
        n_head=2,
        n_embd=16,
        topk_m=3,
        pooling="mean",
        beta_init=0.01,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = PhaseekV3Classifier(config)
    tokens = torch.tensor(
        [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0], [5, 4, 3, 2, 1, 6, 7, 0, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    matrices = torch.randn(2, 3, 12, 12)
    matrices = 0.5 * (matrices + matrices.transpose(-1, -2))
    labels = torch.tensor([0, 1])
    logits, auxiliary = model(tokens, matrices)
    loss = F.cross_entropy(logits, labels) + auxiliary["mixture_regularization"]
    loss.backward()

    assert model.mixer.alpha.grad is not None
    assert model.mixer.delta.grad is not None
    assert torch.isfinite(model.mixer.alpha.grad).all()
    assert torch.isfinite(model.mixer.delta.grad).all()
    assert model.mixer.alpha.grad.norm().item() > 0
    assert model.mixer.delta.grad.norm().item() > 0
    for block in model.blocks:
        assert block.attn.beta.grad is not None
        assert torch.isfinite(block.attn.beta.grad).all()
