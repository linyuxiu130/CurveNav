import numpy as np

from curvenav.data_generation.audit_hssd_v2_selector_gap import (
    _fixed_smoke_rows,
    _rank_comparison,
)


def test_rank_comparison_detects_reversed_progress_order() -> None:
    result = _rank_comparison(
        np.array([3.0, 2.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
        np.ones(3, dtype=bool),
    )

    assert result["comparable_pairs"] == 3
    assert result["concordant_pairs"] == 0
    assert result["pairwise_agreement"] == 0.0
    assert result["spearman"] == -1.0
    assert not result["top1_agreement"]


def test_fixed_smoke_covers_every_scene_and_both_splits() -> None:
    shard_states = {
        **{f"train__scene_{index}.npz": 1000 for index in range(8)},
        **{f"validation__scene_{index}.npz": 1000 for index in range(2)},
    }

    selected = _fixed_smoke_rows(shard_states)

    assert set(selected) == set(shard_states)
    assert sum(len(rows) for rows in selected.values()) == 128
    assert all(len(rows) == len(set(rows)) for rows in selected.values())
    assert sum(
        len(rows) for name, rows in selected.items() if name.startswith("validation__")
    ) == 32
