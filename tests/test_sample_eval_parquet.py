from __future__ import annotations

import pytest

from scripts.sample_eval_parquet import select_indices


def test_select_indices_uses_seeded_stable_hash_and_preserves_source_order():
    assert select_indices(["a", "b", "c", "d"], rows=2, seed=42) == [1, 3]


def test_select_indices_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="unique"):
        select_indices(["a", "a", "b"], rows=2, seed=42)


def test_select_indices_rejects_invalid_row_count():
    with pytest.raises(ValueError, match="between 1 and 2"):
        select_indices(["a", "b"], rows=0, seed=42)
    with pytest.raises(ValueError, match="between 1 and 2"):
        select_indices(["a", "b"], rows=3, seed=42)
