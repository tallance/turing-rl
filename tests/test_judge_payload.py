import os
from shared.api_client import build_chat_payload
from shared.judge_prompts import TURING_RESPONSE_PROPERTIES, TURING_RESPONSE_SCHEMA


def test_sampling_merged():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192,
                           reasoning=False, sampling={"temperature": 0.6, "top_k": 20})
    assert p["temperature"] == 0.6 and p["top_k"] == 20


def test_chat_template_kwargs():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192,
                           reasoning=False, chat_template_kwargs={"enable_thinking": False})
    assert p["chat_template_kwargs"] == {"enable_thinking": False}


def test_defaults_noop():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192, reasoning=False)
    assert "temperature" not in p and "chat_template_kwargs" not in p


def test_disable_openrouter_extras(monkeypatch):
    monkeypatch.setenv("PERSONA_DISABLE_OPENROUTER_EXTRAS", "1")
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192, reasoning=False)
    assert "provider" not in p and "reasoning" not in p


from training.grpo.reward import _resolve_response_format


def test_json_schema_on(monkeypatch):
    monkeypatch.setenv("PERSONA_JUDGE_JSON_SCHEMA", "1")
    rf = _resolve_response_format()
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]["schema"]
    assert schema is TURING_RESPONSE_SCHEMA
    assert list(schema["properties"]) == list(TURING_RESPONSE_PROPERTIES)
    assert schema["required"] == list(TURING_RESPONSE_PROPERTIES)
    assert schema["required"][-1] == "rating"
    assert len(schema["required"]) == 37
    assert schema["additionalProperties"] is False


def test_json_schema_off(monkeypatch):
    monkeypatch.delenv("PERSONA_JUDGE_JSON_SCHEMA", raising=False)
    assert _resolve_response_format() == {"type": "json_object"}
