import pandas as pd

from phaseek_v2.manifest import assign_grouped_splits


def test_grouped_split_has_no_leakage_and_both_classes():
    rows = []
    for label in (0, 1):
        for group_index in range(30):
            for replicate in range(2):
                rows.append(
                    {
                        "sample_id": f"{label}_{group_index}_{replicate}",
                        "label": label,
                        "group_id": f"g_{label}_{group_index}",
                    }
                )
    frame = assign_grouped_splits(pd.DataFrame(rows), seed=123)
    for split in ("train", "val", "test"):
        assert set(frame.loc[frame.split == split, "label"]) == {0, 1}
    group_splits = frame.groupby("group_id")["split"].nunique()
    assert group_splits.max() == 1
