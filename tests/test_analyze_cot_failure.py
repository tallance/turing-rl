from scripts.analyze_cot_failure import (
    repetition_metrics, classify_outcome, extract_call,
)


def test_repetition_separates_loop_from_diverse():
    loop = "Please tell me. Thank you.\n" * 60
    diverse = ("The user asks about Israel. Response A repeats prior text verbatim, a "
               "generation artifact. Response B pivots naturally to demographics with a "
               "human typo. Style, goal, and target all favor B as the real person here.")
    rl = repetition_metrics(loop)
    rd = repetition_metrics(diverse)
    # loop is far more compressible, has fewer distinct trigrams, longer literal run
    assert rl["zlib_ratio"] > rd["zlib_ratio"]
    assert rl["distinct_3gram"] < rd["distinct_3gram"]
    assert rl["max_line_repeat"] >= 50
    assert rd["max_line_repeat"] == 1


def test_repetition_empty_text():
    m = repetition_metrics("")
    assert m["zlib_ratio"] == 1.0 and m["distinct_3gram"] == 1.0 and m["max_line_repeat"] == 0


def test_classify_outcome():
    assert classify_outcome("stop", '{"rating": 3}') == "ok"
    assert classify_outcome("length", '{"rating": 3}') == "ok"     # valid wins over finish
    assert classify_outcome("length", "") == "cap_runaway"
    assert classify_outcome("stop", "no json here") == "stop_malformed"
    assert classify_outcome(None, "") == "timeout"
    assert classify_outcome("stop", '{"rating": 9}') == "stop_malformed"  # out of 1-7 range


def _http_row(reasoning, content, finish, ctok):
    return {"response": {"choices": [{"message": {"reasoning": reasoning, "content": content},
                                      "finish_reason": finish}],
                         "usage": {"completion_tokens": ctok}}}


def test_extract_call_maps_fields():
    c = extract_call(_http_row("some thinking", '{"rating": 5}', "stop", 1234))
    assert c["completion_tokens"] == 1234
    assert c["outcome"] == "ok"
    assert c["finish_reason"] == "stop"
    assert c["reasoning_chars"] == len("some thinking")


def test_extract_call_runaway():
    c = extract_call(_http_row("loop loop loop", "", "length", 8192))
    assert c["outcome"] == "cap_runaway"


def test_extract_call_no_choices():
    assert extract_call({"response": {"choices": []}}) is None
