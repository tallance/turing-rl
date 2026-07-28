# tests/test_rollout_sync_guard.py
import numpy as np
import pytest

from scripts.rollout_sync_guard import (
    assert_fixed_sequence_delta_synced,
    assert_rollout_synced,
    logprob_parity,
)

def test_parity_close_and_far():
    a = np.array([-0.10, -1.20, -0.03])
    assert logprob_parity(a, a + 0.02, atol=0.1)["close"] is True      # within tol
    r = logprob_parity(a, a + 0.5, atol=0.1)
    assert r["close"] is False and r["max_abs_diff"] > 0.4             # rollout != actor

def test_ok_when_synced_both_steps_and_policy_moved():
    # within-step parity dicts (rollout vs actor, same sampled tokens — may differ per step)
    s0 = logprob_parity(np.array([-0.1, -0.2]), np.array([-0.11, -0.19]))   # close @ step0
    s1 = logprob_parity(np.array([-0.4, -0.9, -0.3]), np.array([-0.41, -0.9, -0.29]))  # close @ step1
    # teacher-forced logprobs on a FIXED 4-token continuation, evaluated at both steps (same shape)
    tf0 = np.array([-0.10, -0.20, -0.30, -0.40])
    tf1 = np.array([-0.50, -0.60, -0.70, -0.80])                            # actor moved
    out = assert_rollout_synced(s0, s1, tf0, tf1)
    assert out == {"synced": True, "policy_moved": True, "ok": True}

def test_stale_base_flagged():           # rollout frozen at base while actor trains -> step1 desync
    s0 = logprob_parity(np.array([-0.1, -0.2]), np.array([-0.11, -0.19]))   # ok @ step0
    s1 = logprob_parity(np.array([-0.4, -0.9]), np.array([-0.4, -0.2]))     # rollout != actor @ step1
    tf0 = np.array([-0.1, -0.2, -0.3, -0.4]); tf1 = np.array([-0.5, -0.6, -0.7, -0.8])
    out = assert_rollout_synced(s0, s1, tf0, tf1)
    assert out["synced"] is False and out["ok"] is False

def test_frozen_policy_flagged():        # rollout tracks actor but teacher-forced logprobs unchanged
    s = logprob_parity(np.array([-0.1, -0.2]), np.array([-0.11, -0.19]))
    tf = np.array([-0.1, -0.2, -0.3, -0.4])
    out = assert_rollout_synced(s, s, tf, tf.copy())                        # no movement on fixed seq
    assert out["policy_moved"] is False and out["ok"] is False

def test_teacher_forced_shape_must_match():
    s = logprob_parity(np.array([-0.1]), np.array([-0.1]))
    with pytest.raises((ValueError, AssertionError)):
        assert_rollout_synced(s, s, np.array([-0.1, -0.2]), np.array([-0.1]))  # not a fixed seq


def test_fixed_sequence_delta_ignores_constant_engine_offset():
    actor0 = np.array([-1.0, -2.0, -3.0, -4.0])
    actor1 = np.array([-1.2, -1.7, -3.4, -3.8])
    rollout0 = actor0 + 0.7
    rollout1 = actor1 + 0.7

    out = assert_fixed_sequence_delta_synced(actor0, actor1, rollout0, rollout1)

    assert out["ok"] is True
    assert out["synced"] is True
    assert out["policy_moved"] is True
    assert out["rollout_moved"] is True
    assert out["delta_error_p99_abs"] < 1e-12
    assert out["actor_move_mean_abs"] == pytest.approx(0.275)
    assert out["rollout_move_mean_abs"] == pytest.approx(0.275)
    assert out["actor_move_p99_abs"] > 0.39


def test_fixed_sequence_delta_flags_stale_rollout():
    actor0 = np.array([-1.0, -2.0, -3.0, -4.0])
    actor1 = np.array([-1.2, -1.7, -3.4, -3.8])
    rollout0 = actor0 + 0.7
    rollout1 = rollout0.copy()

    out = assert_fixed_sequence_delta_synced(actor0, actor1, rollout0, rollout1)

    assert out["ok"] is False
    assert out["synced"] is False
    assert out["policy_moved"] is True
    assert out["rollout_moved"] is False


def test_fixed_sequence_delta_requires_identical_shapes():
    with pytest.raises(ValueError, match="same fixed sequence"):
        assert_fixed_sequence_delta_synced(
            np.array([-1.0, -2.0]),
            np.array([-1.1]),
            np.array([-1.0, -2.0]),
            np.array([-1.1, -2.1]),
        )
