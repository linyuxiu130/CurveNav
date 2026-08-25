import torch
from torch import nn

from curvenav.training.ema import ExponentialMovingAverage


def test_ema_warmup_tracks_early_updates_and_roundtrips_state() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ExponentialMovingAverage(model, decay=0.9999)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update()

    assert ema.num_updates == 1
    assert ema.shadow["weight"].item() > 0.8

    restored = ExponentialMovingAverage(model, decay=0.5)
    restored.load_state_dict(ema.state_dict())
    assert restored.decay == ema.decay
    assert restored.num_updates == ema.num_updates
    assert torch.equal(restored.shadow["weight"], ema.shadow["weight"])


def test_fused_ema_matches_parameterwise_lerp_exactly() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
    ema = ExponentialMovingAverage(model, decay=0.9999)
    reference = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    for update in range(1, 101):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter))
        effective_decay = min(ema.decay, (1.0 + update) / (10.0 + update))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                reference[name].lerp_(parameter, 1.0 - effective_decay)
        ema.update()

    for name, shadow in ema.shadow.items():
        assert torch.equal(shadow, reference[name])
