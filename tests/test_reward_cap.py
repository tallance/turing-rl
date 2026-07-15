# tests/test_reward_cap.py
import importlib, os, pytest
import training.grpo.reward as R

def _reload():
    importlib.reload(R)  # pick up module-level default; env is read per-call
    return R

def test_default_cap_is_5(monkeypatch):
    monkeypatch.delenv("TURING_JUDGE_SCORE_CLIP_MAX", raising=False)
    r = _reload()
    assert r.clip_turing_judge_score(7) == 5.0
    assert r.clip_turing_judge_score(4) == 4.0

def test_env_cap_7_is_noop(monkeypatch):
    monkeypatch.setenv("TURING_JUDGE_SCORE_CLIP_MAX", "7")
    r = _reload()
    assert r.clip_turing_judge_score(7) == 7.0
    assert r.clip_turing_judge_score(6) == 6.0

def test_reward_math_cap5_vs_cap7(monkeypatch):
    # unadjusted = (clip-1)/6 ; adjusted = *0.9
    monkeypatch.setenv("TURING_JUDGE_SCORE_CLIP_MAX", "5")
    r = _reload()
    c = r.clip_turing_judge_score(7)
    assert r.adjust_turing_raw_reward((c - 1) / 6) == pytest.approx(0.6)   # (5-1)/6*0.9
    monkeypatch.setenv("TURING_JUDGE_SCORE_CLIP_MAX", "7")
    r = _reload()
    c = r.clip_turing_judge_score(7)
    assert r.adjust_turing_raw_reward((c - 1) / 6) == pytest.approx(0.9)   # (7-1)/6*0.9

def test_bad_env_raises(monkeypatch):
    monkeypatch.setenv("TURING_JUDGE_SCORE_CLIP_MAX", "notafloat")
    r = _reload()
    with pytest.raises(ValueError):
        r.clip_turing_judge_score(7)
