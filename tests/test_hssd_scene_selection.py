from scripts.select_hssd_batch import _family_disjoint_queues


def _record(scene_id: str, family: str) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "source_family": family,
        "objects": 10,
        "regions": 3,
        "region_labels": 2,
        "annotated_floor_area_m2": 30.0,
        "largest_region_m2": 15.0,
    }


def test_hssd_selection_holds_out_complete_scene_families():
    train = [
        _record("forced", "a"),
        _record("train-b", "b"),
        _record("train-c", "c"),
        _record("train-d", "d"),
    ]
    validation = [
        _record("val-a", "a"),
        _record("val-b", "b"),
        _record("val-c-1", "c"),
        _record("val-c-2", "c"),
        _record("val-d", "d"),
    ]

    queues, held_out = _family_disjoint_queues(
        train,
        validation,
        validation_scenes=2,
        forced_train_scene="forced",
    )

    family = {record["scene_id"]: record["source_family"] for record in train + validation}
    train_families = {family[scene] for scene in queues["train"]}
    validation_families = {family[scene] for scene in queues["validation"]}
    assert len(held_out) == 2
    assert validation_families == set(held_out)
    assert train_families.isdisjoint(validation_families)
    assert queues["train"][0] == "forced"
