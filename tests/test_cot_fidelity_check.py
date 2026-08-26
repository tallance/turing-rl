"""Unit tests for the PURE helpers of the CoT fidelity check.

Covers only the network-free comparison helpers (``perspective`` and
``summarize``) with synthetic strings. The two generation backends
(OpenRouter on the Mac, self-hosted on the cluster) are exercised at run
time, not here.
"""

from scripts.cot_fidelity_check import perspective, summarize


def test_perspective_first_person():
    assert perspective("I think the weather is nice") == "first"
    assert perspective("My plan is to reply calmly") == "first"
    assert perspective("I'm annoyed, so me and my friend left") == "first"


def test_perspective_third_person():
    assert perspective("The user wants to vent about the mods") == "third"
    assert perspective("They are frustrated with the reply") == "third"
    assert perspective("The person is being sarcastic here") == "third"


def test_perspective_other():
    assert perspective("") == "other"
    assert perspective("   ") == "other"
    assert perspective("Weather looks cloudy today") == "other"


def test_perspective_earliest_marker_wins():
    # first-person marker appears before any third-person marker -> first
    assert perspective("I can tell the user is upset") == "first"
    # third-person marker appears first -> third
    assert perspective("The user says I should calm down") == "third"


def test_summarize_quartiles_and_counts():
    s = summarize(["ab", "abcd"])
    assert s["n"] == 2
    assert s["empty"] == 0
    assert s["len_p25"] == 2.5
    assert s["len_p50"] == 3.0
    assert s["len_p75"] == 3.5
    assert s["perspective_counts"]["other"] == 2


def test_summarize_empty_and_perspective_mix():
    s = summarize(["I am here", "  ", "The user left"])
    assert s["n"] == 3
    assert s["empty"] == 1
    assert s["perspective_counts"]["first"] == 1
    assert s["perspective_counts"]["third"] == 1
    assert s["perspective_counts"]["other"] == 1


def test_summarize_handles_no_texts():
    s = summarize([])
    assert s["n"] == 0
    assert s["empty"] == 0
    assert s["len_p50"] == 0.0
    assert s["perspective_counts"] == {"first": 0, "third": 0, "other": 0}
