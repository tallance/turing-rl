"""Minimal web viewer for judge-response dumps.

Renders JSONL files from two sources:
  - HTTP-layer dumps written by shared/api_client.py:_dump_judge_response
    (under ${PERSONA_JUDGE_DUMP_DIR}/http/judge-*.jsonl). Shows wire-level
    request/response, useful for debugging HTTP/vLLM issues.
  - Reward-layer dumps written by training/grpo/reward.py:_dump_reward_call
    (under ${PERSONA_JUDGE_DUMP_DIR}/reward/reward-*.jsonl). Shows the
    reward-time context: which side is human (A/B), ground truth, penalty
    breakdown, final reward.

Schema is auto-detected per row: presence of `generated_is_b` field ⇒ reward
row; otherwise HTTP row. The viewer recurses into subdirs, so passing the
parent directory picks up both types.

Usage:
  python scripts/dump_viewer.py --dumps /home/lancewicki/tmp/judge_dumps_8b --port 8082

Access from Mac:
  ssh -L 8082:localhost:8082 <cluster-host>
  # then browse http://localhost:8082
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Template

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from training.grpo.reward import _extract_json  # noqa: E402  # match production parse

# --- data loading ------------------------------------------------------------

HTTP_TABS = ("prompt", "response", "parsed", "reasoning", "metadata")
REWARD_TABS = ("context", "history", "response", "ground_truth", "prompt", "raw", "reasoning", "judge", "reward", "metadata")


def _scan_dumps(dumps_dir: Path) -> pd.DataFrame:
    """Read every *.jsonl under dumps_dir (recursively) into a single DataFrame.

    Auto-detects schema per row: rows with `generated_is_b` are reward-layer
    dumps; rows with `payload_messages` are HTTP-layer dumps. Unknown rows are
    skipped.
    """
    paths = sorted(glob.glob(str(dumps_dir / "**" / "*.jsonl"), recursive=True))
    rows: list[dict] = []
    for path in paths:
        stem = Path(path).stem
        # judge-<slurm>-<pid>  or  reward-<slurm>-<pid>
        job_and_pid = stem.split("-", 1)[1] if "-" in stem else stem
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "generated_is_b" in d:
                    # Reward-layer row.
                    judge = d.get("judge_response") or {}
                    rating = d.get("rating_gt_first") if d.get("rating_gt_first") is not None else d.get("rating_gen_first")
                    human_side = d.get("human_side")
                    # "correct" = 1.0 if the judge picked the actual human side,
                    # 0.0 if it picked the generator side, 0.5 if tie (rating=4).
                    # rating 1..3 => judge picks A; 5..7 => judge picks B; 4 => tie.
                    correct: float | None
                    if rating is None or human_side not in ("A", "B"):
                        correct = None
                    else:
                        try:
                            r_int = int(rating)
                        except (TypeError, ValueError):
                            correct = None
                        else:
                            if r_int == 4:
                                correct = 0.5
                            else:
                                judge_pick = "A" if r_int <= 3 else "B"
                                correct = 1.0 if judge_pick == human_side else 0.0
                    rows.append({
                        "path": path,
                        "line_no": line_no,
                        "schema": "reward",
                        "ts": d.get("ts"),
                        "worker_pid": d.get("worker_pid"),
                        "job_and_pid": job_and_pid,
                        "user_id": d.get("user_id"),
                        "post_id": d.get("post_id"),
                        "target_idx": d.get("target_idx"),
                        # Sidebar columns
                        "final_reward": d.get("final_reward"),
                        "rating": rating,
                        "human_side": human_side,
                        "correct": correct,
                        # Search
                        "_search_blob": (
                            str(d.get("response", "")) + "\n"
                            + str(d.get("ground_truth", "")) + "\n"
                            + str(d.get("context", ""))
                        ).lower(),
                        # Legacy fields kept as None so filters/sort don't blow up.
                        "latency_ms": None,
                        "model": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "has_reasoning": bool((judge or {}).get("reasoning") if isinstance(judge, dict) else False),
                        "parses_ok": judge is not None and bool(judge),
                        "prompt_len": len(str(d.get("context", ""))),
                        "content_len": len(str(d.get("response", ""))),
                        "raw": d,
                    })
                elif "payload_messages" in d or "response" in d:
                    # HTTP-layer row (legacy shape).
                    choice = (d.get("response", {}).get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    usage = d.get("response", {}).get("usage") or {}
                    payload_msgs = d.get("payload_messages") or []
                    prompt_text = "\n\n".join(m.get("content", "") for m in payload_msgs)
                    parses_ok = _extract_json(content) is not None
                    rows.append({
                        "path": path,
                        "line_no": line_no,
                        "schema": "http",
                        "ts": d.get("ts"),
                        "latency_ms": d.get("latency_ms"),
                        "model": d.get("model"),
                        "worker_pid": d.get("worker_pid"),
                        "job_and_pid": job_and_pid,
                        "user_id": None,
                        "post_id": None,
                        "target_idx": None,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "has_reasoning": bool(message.get("reasoning")),
                        "parses_ok": parses_ok,
                        "prompt_len": len(prompt_text),
                        "content_len": len(content),
                        "raw": d,
                        "_search_blob": (prompt_text + "\n" + content).lower(),
                        # Reward-only columns default to None so the DataFrame
                        # is rectangular.
                        "final_reward": None,
                        "rating": None,
                        "human_side": None,
                        "correct": None,
                    })
                # else: unknown schema — skip silently.
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Chronological order by ts (stable sort; rows without a ts sink to the end).
    # This makes idx / sidebar order reflect training-call order across all files.
    df = df.sort_values("ts", kind="stable", na_position="last").reset_index(drop=True)
    # Per-example chronological sequence number (1-based). Reward rows carry the
    # example identity (user_id/post_id/target_idx), so this lets you step through
    # one example's generations + judge calls in training order.
    df["seq"] = df.groupby(["user_id", "post_id", "target_idx"], dropna=False).cumcount() + 1
    df["idx"] = df.index
    return df


# --- app state ---------------------------------------------------------------

class State:
    dumps_dir: Path
    df: pd.DataFrame

    def __init__(self, dumps_dir: Path) -> None:
        self.dumps_dir = dumps_dir
        self.reload()

    def reload(self) -> None:
        self.df = _scan_dumps(self.dumps_dir)


state: State | None = None  # set in main()


# --- app ---------------------------------------------------------------------

app = FastAPI(title="Judge dump viewer")

TEMPLATE = Template("""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>judge dumps</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; color: #222; }
  .layout { display: flex; height: 100vh; }
  .sidebar { width: 30%; min-width: 260px; max-width: 500px; border-right: 1px solid #ddd; display: flex; flex-direction: column; }
  .filters { padding: 8px; border-bottom: 1px solid #ddd; background: #fafafa; }
  .filters form { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
  .filters input[type=text], .filters input[type=number] { width: 80px; padding: 2px 4px; font-size: 12px; }
  .filters input[name=q] { width: 160px; }
  .filters label { font-size: 11px; color: #555; }
  .filters button { padding: 2px 8px; font-size: 12px; cursor: pointer; }
  .counts { padding: 4px 8px; background: #f4f4f4; border-bottom: 1px solid #ddd; font-size: 11px; color: #555; }
  .rows { flex: 1; overflow-y: auto; }
  table.rows-table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
  table.rows-table th, table.rows-table td { padding: 3px 6px; text-align: left; border-bottom: 1px solid #f0f0f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  table.rows-table th { background: #eee; position: sticky; top: 0; font-weight: 600; }
  table.rows-table col.idx { width: 42px; }
  table.rows-table col.tag { width: 60px; }
  table.rows-table col.num { width: 60px; }
  table.rows-table col.short { width: 44px; }
  table.rows-table tr { cursor: pointer; }
  table.rows-table tr:hover { background: #fff8e1; }
  table.rows-table tr.selected { background: #fef3c7; font-weight: 600; }
  table.rows-table td.right, table.rows-table th.right { text-align: right; font-variant-numeric: tabular-nums; }
  table.rows-table td.ok { color: #087f23; }
  table.rows-table td.bad { color: #b71c1c; font-weight: 600; }
  .schema-tag { display: inline-block; padding: 0 4px; margin-right: 4px; border-radius: 2px; font-size: 10px; font-weight: 600; color: white; }
  .schema-tag.reward { background: #1976d2; }
  .schema-tag.http { background: #888; }
  .human-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-weight: 600; color: white; background: #087f23; }
  a { color: inherit; text-decoration: none; }
  .detail { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .tabs { display: flex; border-bottom: 1px solid #ddd; background: #fafafa; }
  .tabs a { padding: 8px 14px; border-right: 1px solid #ddd; color: #555; }
  .tabs a.active { background: white; color: #000; font-weight: 600; border-bottom: 2px solid #1976d2; margin-bottom: -1px; }
  .tab-body { flex: 1; overflow: auto; padding: 10px 14px; }
  .tab-body pre { white-space: pre-wrap; word-wrap: break-word; margin: 0; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; line-height: 1.4; }
  .meta-grid { display: grid; grid-template-columns: 180px 1fr; gap: 4px 12px; font-size: 12px; }
  .meta-grid dt { color: #555; }
  .meta-grid dd { margin: 0; font-family: "SF Mono", Menlo, monospace; }
  .header-bar { padding: 6px 14px; background: #f4f4f4; border-bottom: 1px solid #ddd; font-size: 12px; color: #555; }
  .parse-error { color: #b71c1c; padding: 6px; background: #fef2f2; border-left: 3px solid #b71c1c; margin-bottom: 8px; }
  .empty { padding: 40px; text-align: center; color: #999; }
</style>
</head>
<body>
<div class="layout">

  <div class="sidebar">
    <div class="filters">
      <form method="get" action="/">
        <label>text: <input type="text" name="q" value="{{ filters.q or '' }}" placeholder="substring"></label>
        <label>user_id: <input type="text" name="user_id" value="{{ filters.user_id if filters.user_id is not none else '' }}" placeholder="exact"></label>
        <label>post_id: <input type="text" name="post_id" value="{{ filters.post_id if filters.post_id is not none else '' }}" placeholder="exact"></label>
        <label>target_idx: <input type="text" name="target_idx" value="{{ filters.target_idx if filters.target_idx is not none else '' }}" placeholder="exact"></label>
        <label>schema:
          <select name="schema">
            <option value="all" {% if filters.schema == 'all' %}selected{% endif %}>all</option>
            <option value="reward" {% if not filters.schema or filters.schema == 'reward' %}selected{% endif %}>reward</option>
            <option value="http" {% if filters.schema == 'http' %}selected{% endif %}>http</option>
          </select>
        </label>
        <label><input type="checkbox" name="only_parse_failures" value="1" {% if filters.only_parse_failures %}checked{% endif %}> failed parses</label>
        <button type="submit">apply</button>
        <a href="/?reload=1" style="margin-left: 8px; color: #1976d2;">↻ reload</a>
      </form>
    </div>
    <div class="counts">
      {{ visible_count }} / {{ total_count }} rows
      {% if visible_count > page_size %}(showing first {{ page_size }}){% endif %}
    </div>
    <div class="rows">
      {% if visible_count == 0 %}
      <div class="empty">no rows match</div>
      {% else %}
      <table class="rows-table">
        <colgroup>
          <col class="idx">
          {% if filters.schema == 'all' %}<col class="tag">{% endif %}
          {% if filters.schema == 'http' %}
            <col class="num"><col class="num"><col class="short">
          {% else %}
            <col class="short"><col class="num"><col class="short"><col class="short"><col class="short">
          {% endif %}
        </colgroup>
        <thead><tr>
          <th>idx</th>
          {% if filters.schema == 'all' %}
            <th></th>
          {% endif %}
          {% if filters.schema == 'http' %}
            <th class="right">lat(ms)</th>
            <th class="right">in→out</th>
            <th>ok</th>
          {% else %}
            <th class="right">seq</th>
            <th class="right">reward</th>
            <th class="right">rating</th>
            <th>human</th>
            <th>correct</th>
          {% endif %}
        </tr></thead>
        <tbody>
        {% for row in visible_rows %}
          <tr class="{% if row.idx == selected_idx %}selected{% endif %}"
              onclick="window.location.href='{{ row.link }}'">
            <td>{{ row.idx }}</td>
            {% if filters.schema == 'all' %}
              <td><span class="schema-tag {{ row.schema }}">{{ row.schema }}</span></td>
            {% endif %}
            {% if filters.schema == 'http' %}
              <td class="right">{{ '%.0f' % row.latency_ms if row.latency_ms is not none else '-' }}</td>
              <td class="right">{{ row.prompt_tokens or '-' }} → {{ row.completion_tokens or '-' }}</td>
              <td class="{{ 'ok' if row.parses_ok else 'bad' }}">{{ '✓' if row.parses_ok else '✗' }}</td>
            {% else %}
              <td class="right">{{ row.seq if row.seq is not none else '-' }}</td>
              <td class="right">{{ '%.2f' % row.final_reward if row.final_reward is not none else '-' }}</td>
              <td class="right">{{ row.rating|int if row.rating is not none else '-' }}</td>
              <td>{{ row.human_side or '-' }}</td>
              <td class="{{ 'ok' if row.correct == 1.0 else ('bad' if row.correct == 0.0 else '') }}">{% if row.correct is none %}-{% elif row.correct == 1.0 %}1{% elif row.correct == 0.0 %}0{% else %}½{% endif %}</td>
            {% endif %}
          </tr>
        {% endfor %}
        </tbody>
      </table>
      {% endif %}
    </div>
  </div>

  <div class="detail">
    {% if selected is none %}
    <div class="empty" style="margin: auto;">select a row on the left</div>
    {% else %}
    <div class="header-bar">
      <span class="schema-tag {{ selected.schema }}">{{ selected.schema }}</span>
      idx <strong>{{ selected.idx }}</strong>
      &nbsp;·&nbsp; {{ selected.human_ts }}
      {% if selected.schema == 'reward' %}
        &nbsp;·&nbsp; <span class="human-badge">Human: {{ selected.human_side }}</span>
        &nbsp;·&nbsp; reward = <strong>{{ '%.3f' % selected.final_reward if selected.final_reward is not none else '-' }}</strong>
        &nbsp;·&nbsp; rating = <strong>{{ selected.rating if selected.rating is not none else '-' }}</strong>
        &nbsp;·&nbsp; ordering: {{ selected.randomized_order }}
      {% else %}
        &nbsp;·&nbsp; {{ selected.model }}
        &nbsp;·&nbsp; {{ '%.0f ms' % selected.latency_ms if selected.latency_ms is not none else 'n/a ms' }}
        &nbsp;·&nbsp; {{ selected.prompt_tokens or '-' }} → {{ selected.completion_tokens or '-' }} tokens
        {% if selected.parses_ok %}<span class="ok">· parses ✓</span>{% else %}<span style="color:#b71c1c;">· parse ✗</span>{% endif %}
      {% endif %}
      &nbsp;·&nbsp; <a href="/raw/{{ selected.idx }}" target="_blank" style="color: #1976d2;">raw json</a>
    </div>
    <div class="tabs">
      {% for tab_name in tab_names %}
      <a href="{{ selected.tab_links[tab_name] }}" class="{% if tab_name == active_tab %}active{% endif %}">{{ tab_name }}</a>
      {% endfor %}
    </div>
    <div class="tab-body">
      {% if selected.schema == 'reward' %}
        {% if active_tab == 'context' %}
          <pre>{{ selected.context_text }}</pre>
        {% elif active_tab == 'history' %}
          <pre>{{ selected.user_history }}</pre>
        {% elif active_tab == 'response' %}
          <div style="margin-bottom: 8px; color: #555;">Generator's output (human is on side <strong>{{ selected.human_side }}</strong>, so this is on side <strong>{{ 'B' if selected.human_side == 'A' else 'A' }}</strong>):</div>
          <pre>{{ selected.response_text }}</pre>
        {% elif active_tab == 'ground_truth' %}
          <div style="margin-bottom: 8px; color: #555;">Actual human response (side <strong>{{ selected.human_side }}</strong>):</div>
          <pre>{{ selected.ground_truth }}</pre>
        {% elif active_tab == 'prompt' %}
          <div style="margin-bottom: 8px; color: #555;">Assembled Turing prompt actually sent to the judge:</div>
          {% if selected.judge_prompt %}
            <pre>{{ selected.judge_prompt }}</pre>
          {% else %}
            <div class="empty">no judge_prompt on this row (older dump before enrichment)</div>
          {% endif %}
        {% elif active_tab == 'raw' %}
          <div style="margin-bottom: 8px; color: #555;">Raw judge response content (before JSON extraction):</div>
          {% if selected.judge_raw_content %}
            <pre>{{ selected.judge_raw_content }}</pre>
          {% else %}
            <div class="empty">no judge_raw_content on this row (older dump before enrichment)</div>
          {% endif %}
        {% elif active_tab == 'reasoning' %}
          <div style="margin-bottom: 8px; color: #555;">Judge's chain-of-thought (&lt;think&gt; contents):</div>
          {% if selected.judge_reasoning %}
            <pre>{{ selected.judge_reasoning }}</pre>
          {% else %}
            <div class="empty">no judge_reasoning on this row</div>
          {% endif %}
        {% elif active_tab == 'judge' %}
          {% if selected.judge_pretty %}
            <pre>{{ selected.judge_pretty }}</pre>
          {% else %}
            <div class="empty">no judge_response on this row</div>
          {% endif %}
        {% elif active_tab == 'reward' %}
          <dl class="meta-grid">
            <dt>final_reward</dt><dd>{{ selected.final_reward }}</dd>
            <dt>turing_judge_score_raw</dt><dd>{{ selected.turing_judge_score_raw }}</dd>
            <dt>turing_judge_score_clipped</dt><dd>{{ selected.turing_judge_score_clipped }}</dd>
            <dt>rating_gt_first</dt><dd>{{ selected.rating_gt_first }}</dd>
            <dt>rating_gen_first</dt><dd>{{ selected.rating_gen_first }}</dd>
            <dt>generated_is_b</dt><dd>{{ selected.generated_is_b }}</dd>
            <dt>human_side</dt><dd>{{ selected.human_side }}</dd>
            <dt>randomized_order</dt><dd>{{ selected.randomized_order }}</dd>
            <dt>source_copy_penalty</dt><dd>{{ selected.source_copy_penalty }}</dd>
            <dt>assistant_like_penalty</dt><dd>{{ selected.assistant_like_penalty }}</dd>
            <dt>wrong_target_or_role_penalty</dt><dd>{{ selected.wrong_target_or_role_penalty }}</dd>
            <dt>unsupported_adversarial_reframing_penalty</dt><dd>{{ selected.unsupported_adversarial_reframing_penalty }}</dd>
          </dl>
        {% elif active_tab == 'metadata' %}
          <dl class="meta-grid">
            <dt>call_id</dt><dd>{{ selected.call_id }}</dd>
            <dt>user_id</dt><dd>{{ selected.user_id }}</dd>
            <dt>post_id</dt><dd>{{ selected.post_id }}</dd>
            <dt>target_idx</dt><dd>{{ selected.target_idx }}</dd>
            <dt>seq (chronological, per example)</dt><dd>{{ selected.seq if selected.seq is not none else '-' }}</dd>
            <dt>persona</dt><dd>{{ selected.persona }}</dd>
            <dt>ts (epoch)</dt><dd>{{ selected.ts }}</dd>
            <dt>ts (utc)</dt><dd>{{ selected.human_ts }}</dd>
            <dt>worker_pid</dt><dd>{{ selected.worker_pid }}</dd>
            <dt>job/pid file</dt><dd>{{ selected.job_and_pid }}</dd>
            <dt>judge model</dt><dd>{{ selected.judge_model or '-' }}</dd>
            <dt>judge latency (ms)</dt><dd>{{ selected.judge_latency_ms if selected.judge_latency_ms is not none else '-' }}</dd>
            <dt>finish reason</dt><dd>{{ selected.judge_finish_reason or '-' }}</dd>
            <dt>usage</dt><dd>{{ selected.judge_usage }}</dd>
          </dl>
        {% endif %}
      {% else %}
        {# HTTP schema — original tabs #}
        {% if active_tab == 'prompt' %}
          <pre>{{ selected.prompt_text }}</pre>
        {% elif active_tab == 'response' %}
          <pre>{{ selected.response_content }}</pre>
        {% elif active_tab == 'parsed' %}
          {% if selected.parsed is not none %}
            <pre>{{ selected.parsed_pretty }}</pre>
          {% else %}
            <div class="parse-error">could not extract JSON from response (using training/grpo/reward.py:_extract_json)</div>
            <pre>{{ selected.response_content }}</pre>
          {% endif %}
        {% elif active_tab == 'reasoning' %}
          {% if selected.reasoning %}
            <pre>{{ selected.reasoning }}</pre>
          {% else %}
            <div class="empty">no reasoning field on this response</div>
          {% endif %}
        {% elif active_tab == 'metadata' %}
          <dl class="meta-grid">
            <dt>ts (epoch)</dt><dd>{{ selected.ts }}</dd>
            <dt>ts (utc)</dt><dd>{{ selected.human_ts }}</dd>
            <dt>latency_ms</dt><dd>{{ selected.latency_ms }}</dd>
            <dt>model</dt><dd>{{ selected.model }}</dd>
            <dt>worker_pid</dt><dd>{{ selected.worker_pid }}</dd>
            <dt>job/pid file</dt><dd>{{ selected.job_and_pid }}</dd>
            <dt>prompt_tokens</dt><dd>{{ selected.prompt_tokens }}</dd>
            <dt>completion_tokens</dt><dd>{{ selected.completion_tokens }}</dd>
            <dt>total_tokens</dt><dd>{{ selected.total_tokens }}</dd>
            <dt>prompt chars</dt><dd>{{ selected.prompt_len }}</dd>
            <dt>content chars</dt><dd>{{ selected.content_len }}</dd>
            <dt>parses_ok</dt><dd>{{ selected.parses_ok }}</dd>
            <dt>has_reasoning</dt><dd>{{ selected.has_reasoning }}</dd>
          </dl>
        {% endif %}
      {% endif %}
    </div>
    {% endif %}
  </div>

</div>
</body>
</html>
""", autoescape=True)


def _filter_df(
    df: pd.DataFrame,
    *,
    only_parse_failures: bool,
    q: str | None,
    schema: str | None,
    user_id: str | None = None,
    post_id: str | None = None,
    target_idx: str | None = None,
) -> pd.DataFrame:
    out = df
    if schema in ("http", "reward"):
        out = out[out["schema"] == schema]
    if only_parse_failures:
        out = out[~out["parses_ok"]]
    # Exact per-example filters (compare as strings so int/str target_idx both match).
    if user_id not in (None, ""):
        out = out[out["user_id"].astype("string") == str(user_id)]
    if post_id not in (None, ""):
        out = out[out["post_id"].astype("string") == str(post_id)]
    if target_idx not in (None, ""):
        out = out[out["target_idx"].astype("string") == str(target_idx)]
    if q:
        needle = q.lower()
        out = out[out["_search_blob"].str.contains(needle, regex=False)]
    return out


def _query_string(**kwargs: Any) -> str:
    parts = []
    for k, v in kwargs.items():
        if v is None or v == "" or v is False:
            continue
        if isinstance(v, bool):
            v = "1"
        parts.append(f"{k}={v}")
    return "?" + "&".join(parts) if parts else ""


@app.get("/", response_class=HTMLResponse)
def index(
    idx: int | None = None,
    tab: str | None = None,
    only_parse_failures: bool = False,
    q: str | None = None,
    schema: str | None = "reward",
    user_id: str | None = None,
    post_id: str | None = None,
    target_idx: str | None = None,
    reload: bool = False,
) -> HTMLResponse:
    assert state is not None
    if reload:
        state.reload()

    filters = dict(
        only_parse_failures=only_parse_failures,
        q=q,
        schema=schema,
        user_id=user_id,
        post_id=post_id,
        target_idx=target_idx,
    )
    df_all = state.df
    if df_all.empty:
        html = TEMPLATE.render(
            visible_rows=[], visible_count=0, total_count=0, page_size=200,
            selected=None, selected_idx=None, active_tab=tab or "response",
            tab_names=HTTP_TABS, filters=filters,
        )
        return HTMLResponse(html)

    df_filt = _filter_df(df_all, **filters)
    page_size = 200
    df_page = df_filt.head(page_size)

    def _link_for(row_idx: int, tab_name: str | None = None) -> str:
        return "/" + _query_string(
            idx=row_idx, tab=tab_name,
            **{k: v for k, v in filters.items() if v},
        )

    visible_rows = [
        {
            "idx": int(r["idx"]),
            "schema": r["schema"],
            "latency_ms": r["latency_ms"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "parses_ok": bool(r["parses_ok"]),
            "final_reward": r["final_reward"],
            "rating": r["rating"],
            "human_side": r["human_side"],
            "correct": r["correct"],
            "seq": (int(r["seq"]) if r.get("seq") is not None and pd.notna(r["seq"]) else None),
            "link": _link_for(int(r["idx"])),
        }
        for _, r in df_page.iterrows()
    ]

    selected = None
    active_tab = tab or "response"
    tab_names: tuple[str, ...] = HTTP_TABS
    if idx is not None and 0 <= idx < len(df_all):
        row = df_all.iloc[idx].to_dict()
        raw = row["raw"]
        ts_val = row.get("ts")
        try:
            human_ts = datetime.utcfromtimestamp(float(ts_val)).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            human_ts = "n/a"
        row_schema = row.get("schema", "http")
        tab_names = REWARD_TABS if row_schema == "reward" else HTTP_TABS
        # default tab if none specified
        if tab is None:
            active_tab = tab_names[0]

        if row_schema == "reward":
            judge_response = raw.get("judge_response") or {}
            selected = {
                "idx": int(row["idx"]),
                "schema": "reward",
                "ts": ts_val,
                "human_ts": human_ts,
                "seq": (int(row["seq"]) if row.get("seq") is not None and pd.notna(row.get("seq")) else None),
                "worker_pid": row.get("worker_pid"),
                "job_and_pid": row.get("job_and_pid"),
                # Reward-layer fields
                "call_id": raw.get("call_id"),
                "user_id": raw.get("user_id"),
                "post_id": raw.get("post_id"),
                "target_idx": raw.get("target_idx"),
                "persona": raw.get("persona"),
                "generated_is_b": raw.get("generated_is_b"),
                "human_side": raw.get("human_side"),
                "randomized_order": raw.get("randomized_order"),
                "response_text": raw.get("response", ""),
                "ground_truth": raw.get("ground_truth", ""),
                "context_text": raw.get("context", ""),
                "user_history": raw.get("user_history", ""),
                "final_reward": raw.get("final_reward"),
                "turing_judge_score_raw": raw.get("turing_judge_score_raw"),
                "turing_judge_score_clipped": raw.get("turing_judge_score_clipped"),
                "source_copy_penalty": raw.get("source_copy_penalty"),
                "assistant_like_penalty": raw.get("assistant_like_penalty"),
                "wrong_target_or_role_penalty": raw.get("wrong_target_or_role_penalty"),
                "unsupported_adversarial_reframing_penalty": raw.get("unsupported_adversarial_reframing_penalty"),
                "rating_gt_first": raw.get("rating_gt_first"),
                "rating_gen_first": raw.get("rating_gen_first"),
                "rating": raw.get("rating_gt_first") if raw.get("rating_gt_first") is not None else raw.get("rating_gen_first"),
                "judge_pretty": json.dumps(judge_response, indent=2, ensure_ascii=False) if judge_response else None,
                # Raw judge fields (enriched dump only; older rows have these as "")
                "judge_prompt": raw.get("judge_prompt", ""),
                "judge_raw_content": raw.get("judge_raw_content", ""),
                "judge_reasoning": raw.get("judge_reasoning", ""),
                "judge_latency_ms": raw.get("judge_latency_ms"),
                "judge_finish_reason": raw.get("judge_finish_reason"),
                "judge_model": raw.get("judge_model"),
                "judge_usage": raw.get("judge_usage") or {},
                "tab_links": {
                    name: _link_for(int(row["idx"]), name) for name in REWARD_TABS
                },
            }
        else:
            message = (raw.get("response", {}).get("choices") or [{}])[0].get("message") or {}
            content = message.get("content") or ""
            parsed = _extract_json(content)
            prompt_text = "\n\n".join(m.get("content", "") for m in raw.get("payload_messages", []))
            selected = {
                "idx": int(row["idx"]),
                "schema": "http",
                "ts": ts_val,
                "human_ts": human_ts,
                "latency_ms": row.get("latency_ms"),
                "model": row.get("model"),
                "worker_pid": row.get("worker_pid"),
                "job_and_pid": row.get("job_and_pid"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "total_tokens": row.get("total_tokens"),
                "prompt_len": row.get("prompt_len"),
                "content_len": row.get("content_len"),
                "parses_ok": bool(row.get("parses_ok")),
                "has_reasoning": bool(row.get("has_reasoning")),
                "prompt_text": prompt_text,
                "response_content": content,
                "reasoning": message.get("reasoning") or "",
                "parsed": parsed,
                "parsed_pretty": json.dumps(parsed, indent=2, ensure_ascii=False) if parsed is not None else None,
                "tab_links": {
                    name: _link_for(int(row["idx"]), name) for name in HTTP_TABS
                },
            }

    html = TEMPLATE.render(
        visible_rows=visible_rows,
        visible_count=len(df_filt),
        total_count=len(df_all),
        page_size=page_size,
        selected=selected,
        selected_idx=idx if idx is not None else -1,
        active_tab=active_tab,
        tab_names=tab_names,
        filters=filters,
    )
    return HTMLResponse(html)


@app.get("/raw/{idx}")
def raw(idx: int) -> JSONResponse:
    assert state is not None
    if not (0 <= idx < len(state.df)):
        return JSONResponse({"error": "idx out of range"}, status_code=404)
    return JSONResponse(state.df.iloc[idx]["raw"])


# --- main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", type=Path, default=Path("/home/lancewicki/tmp/judge_dumps"),
                        help="directory containing *.jsonl files (recurses into subdirs; picks up both http/ and reward/)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1 = localhost only)")
    args = parser.parse_args()

    if not args.dumps.is_dir():
        print(f"ERROR: dumps dir does not exist: {args.dumps}", file=sys.stderr)
        sys.exit(2)

    global state
    state = State(args.dumps)
    n_reward = int((state.df["schema"] == "reward").sum()) if not state.df.empty else 0
    n_http = int((state.df["schema"] == "http").sum()) if not state.df.empty else 0
    print(f"loaded {len(state.df)} rows from {args.dumps}  (reward={n_reward}, http={n_http})", flush=True)
    print(f"serving on http://{args.host}:{args.port}/ (Ctrl-C to stop)", flush=True)
    print(f"from your mac:  ssh -L {args.port}:localhost:{args.port} <cluster-host>", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
