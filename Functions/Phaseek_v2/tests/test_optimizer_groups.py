from phaseek_v2.config import ModelConfig
from phaseek_v2.model import PhaseekV2Classifier


def test_graph_parameters_are_not_weight_decayed_and_have_own_group():
    model = PhaseekV2Classifier(
        ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16, topk_m=2)
    )
    groups = model.optimizer_groups(0.01, base_lr=2e-4, graph_lr_multiplier=5.0)
    by_name = {group["group_name"]: group for group in groups}
    graph_ids = {id(parameter) for parameter in by_name["graph"]["params"]}
    assert by_name["graph"]["weight_decay"] == 0.0
    assert by_name["graph"]["lr"] == 1e-3
    assert id(model.mixer.alpha) in graph_ids
    assert id(model.mixer.delta) in graph_ids
    assert id(model.blocks[0].attn.beta) in graph_ids
