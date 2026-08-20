import torch

from curvenav.models.field import FourierTimeEmbedding


def test_time_embedding_resolves_the_unit_flow_interval() -> None:
    embedding = FourierTimeEmbedding(model_dim=32)
    assert embedding.frequencies[0] == 1.0
    assert torch.isclose(embedding.frequencies[-1], torch.tensor(1e-4))

    times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    values = embedding(times)
    assert values.shape == (5, 32)
    assert not torch.allclose(values[0], values[1])
    assert not torch.allclose(values[1], values[2])
