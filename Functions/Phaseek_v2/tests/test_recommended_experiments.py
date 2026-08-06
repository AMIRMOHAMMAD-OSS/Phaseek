import torch
import torch.nn.functional as F

from phaseek_v2.config import ModelConfig
from phaseek_v2.model import LayerwiseHeadMixture, PhaseekV3Classifier


def small_batch(topk: int = 3):
    tokens = torch.tensor(
        [[1, 2, 3, 4, 0, 0], [4, 3, 2, 1, 5, 0]],
        dtype=torch.long,
    )
    matrices = torch.randn(2, topk, 6, 6)
    matrices = 0.5 * (matrices + matrices.transpose(-1, -2))
    labels = torch.tensor([0, 1])
    return tokens, matrices, labels


def test_layerwise_mixture_shape_and_gradient():
    torch.manual_seed(11)
    config = ModelConfig(
        block_size=6,
        n_layer=2,
        n_head=2,
        n_embd=16,
        topk_m=3,
        graph_mixer="layerwise",
        mixture_init_std=0.01,
        beta_init=0.05,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = PhaseekV3Classifier(config)
    assert isinstance(model.mixer, LayerwiseHeadMixture)
    tokens, matrices, labels = small_batch()
    logits, auxiliary = model(tokens, matrices)
    assert logits.shape == (2, 2)
    assert auxiliary["mixture_weights"].shape == (2, 2, 3)
    assert torch.allclose(
        auxiliary["mixture_weights"].sum(dim=-1),
        torch.ones(2, 2),
        atol=1e-6,
    )
    loss = F.cross_entropy(logits, labels) + auxiliary["mixture_regularization"]
    loss.backward()
    assert model.mixer.logits.grad is not None
    assert torch.isfinite(model.mixer.logits.grad).all()
    assert model.mixer.logits.grad.norm().item() > 0


def test_graph_optimizer_group_has_lr_multiplier():
    model = PhaseekV3Classifier(
        ModelConfig(block_size=6, n_layer=2, n_head=2, n_embd=16, topk_m=3)
    )
    groups = model.optimizer_groups(0.01, base_lr=2e-4, graph_lr_multiplier=5.0)
    by_name = {group["group_name"]: group for group in groups}
    assert by_name["decay"]["lr"] == 2e-4
    assert by_name["no_decay"]["lr"] == 2e-4
    assert by_name["graph"]["lr"] == 1e-3
    graph_ids = {id(parameter) for parameter in by_name["graph"]["params"]}
    assert id(model.mixer.alpha) in graph_ids
    assert id(model.mixer.delta) in graph_ids
    assert id(model.blocks[0].attn.beta) in graph_ids


def test_backbone_freeze_keeps_graph_and_head_trainable():
    model = PhaseekV3Classifier(
        ModelConfig(block_size=6, n_layer=2, n_head=2, n_embd=16, topk_m=3)
    )
    model.set_backbone_frozen(True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert "mixer.alpha" in trainable
    assert "mixer.delta" in trainable
    assert "blocks.0.attn.beta" in trainable
    assert "classifier.weight" in trainable
    assert "pooler.query" in trainable
    assert "token_embedding.weight" not in trainable
    assert "blocks.0.attn.c_attn.weight" not in trainable

    model.set_backbone_frozen(False)
    assert all(parameter.requires_grad for parameter in model.parameters())
