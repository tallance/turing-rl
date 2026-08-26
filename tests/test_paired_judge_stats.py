import pytest

from scripts.paired_judge_stats import clustered_ci, mcnemar


def test_mcnemar_against_a_hand_computed_table():
    # arm A right / arm B wrong: 20.  arm A wrong / arm B right: 5.  Concordant: 100.
    a = [1] * 20 + [0] * 5 + [1] * 100
    b = [0] * 20 + [1] * 5 + [1] * 100
    r = mcnemar(a, b)
    assert r["n_discordant"] == 25
    assert r["b01"] == 20 and r["b10"] == 5
    # chi2 with continuity correction = (|20-5|-1)^2 / 25 = 7.84
    assert r["chi2"] == pytest.approx(7.84, abs=1e-2)
    assert r["p_value"] < 0.01


def test_mcnemar_is_symmetric_in_its_verdict():
    a, b = [1, 0, 1, 1], [0, 1, 1, 1]
    assert mcnemar(a, b)["p_value"] == pytest.approx(mcnemar(b, a)["p_value"])


def test_no_discordant_pairs_is_not_significant():
    assert mcnemar([1, 1, 0], [1, 1, 0])["p_value"] == 1.0


def test_clustering_widens_the_interval_when_orders_agree():
    """Both orders of each pair identical -> effective n is halved, so the clustered
    interval must be materially wider. This widening is the whole reason the column
    exists."""
    correct = [1, 1, 0, 0] * 55          # 110 pairs x 2 orders = 220 rows
    pair_ids = [f"p{i // 2}" for i in range(len(correct))]
    naive = clustered_ci(correct, pair_ids=list(range(len(correct))), seed=0)
    clustered = clustered_ci(correct, pair_ids=pair_ids, seed=0)
    assert clustered["width"] > naive["width"] * 1.2
