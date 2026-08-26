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
    # Asymmetric on purpose: b01=2, b10=3 forward (and swapped, 3, 2), so a formula that
    # is not genuinely symmetric in (b01, b10) has something to get wrong here. A
    # balanced fixture (b01 == b10) can't expose that class of bug.
    a, b = [1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 1]
    fwd, swap = mcnemar(a, b), mcnemar(b, a)
    assert fwd["b01"] == 2 and fwd["b10"] == 3
    assert fwd["b01"] != fwd["b10"]  # guards against a future edit silently rebalancing this
    assert fwd["b01"] == swap["b10"] and fwd["b10"] == swap["b01"]
    assert fwd["p_value"] == pytest.approx(swap["p_value"])


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
