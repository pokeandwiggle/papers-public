import pytest

from paw_baseline.cupid.select import cupid_keep, num_filtered, random_keep


def test_num_filtered_matches_reference_truncation():
    assert num_filtered(308, 0.5) == 154
    assert num_filtered(353, 0.5) == 176
    assert num_filtered(10, 0.66) == 6
    with pytest.raises(ValueError):
        num_filtered(10, 1.0)


def test_random_keep_is_deterministic_and_sized():
    ids = list(range(100, 151))
    first = random_keep(ids, 0.5, seed=0)
    assert first == random_keep(list(reversed(ids)), 0.5, seed=0)
    assert len(first) == 51 - 25
    assert set(first) <= set(ids)
    assert first != random_keep(ids, 0.5, seed=1)


def test_cupid_keep_drops_lowest_scores():
    scores = {1: 0.3, 2: -2.0, 3: 5.0, 4: -0.1, 5: 1.0}
    assert cupid_keep(scores, 0.4) == [1, 3, 5]
    # Ties break by source ID: the smaller ID is dropped first.
    assert cupid_keep({7: 0.0, 8: 0.0, 9: 1.0}, 0.34) == [8, 9]
