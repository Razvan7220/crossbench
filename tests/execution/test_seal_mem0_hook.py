"""Unit tests for the mem0 LiteLLM hook + seal.py integration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cross_bench.execution.scaffolds import mem0_litellm_hook
from cross_bench.execution.scaffolds.mem0_litellm_hook import (
    ENV_DISABLE_STORE,
    ENV_EMB_MODEL,
    ENV_EMB_PROVIDER,
    ENV_FIRST_STEP_ONLY,
    ENV_INFER,
    ENV_LLM_BASE_URL,
    ENV_LLM_MODEL,
    ENV_LLM_PROVIDER,
    ENV_METRICS_FILE,
    ENV_TOP_K,
    ENV_USER_ID,
    SITECUSTOMIZE_STUB_MEM0,
    _build_mem0_config,
    _extract_response_text,
    _flatten_content,
    _format_memories_block,
    _hook_enabled,
    _infer_flag,
    _prepend_to_last_user,
    _recall_query,
    _top_k,
)
from cross_bench.execution.scaffolds.seal import (
    _aggregate_mem0_metrics,
    _build_subprocess_env,
    _resolve_mem0_hook_config,
)


# ---------------------------------------------------------------------------
# Module-state reset fixture (singletons are dangerous in test isolation)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_mem0_module_state():
    mem0_litellm_hook._memory_instance = None
    mem0_litellm_hook._recall_counter[0] = 0
    mem0_litellm_hook._init_logged[0] = False
    # Strip mem0 env vars from previous tests
    for k in list(os.environ.keys()):
        if k.startswith("CROSSBENCH_MEM0_"):
            del os.environ[k]
    yield


# ---------------------------------------------------------------------------
# Env-gate (hook_enabled, top_k, infer_flag)
# ---------------------------------------------------------------------------


def test_hook_disabled_when_user_id_unset():
    assert _hook_enabled() is False


def test_hook_enabled_when_user_id_set():
    os.environ[ENV_USER_ID] = "task-42"
    assert _hook_enabled() is True


def test_hook_disabled_when_user_id_empty_string():
    os.environ[ENV_USER_ID] = ""
    assert _hook_enabled() is False


def test_top_k_default():
    assert _top_k() == mem0_litellm_hook.DEFAULT_TOP_K


def test_top_k_from_env():
    os.environ[ENV_TOP_K] = "7"
    assert _top_k() == 7


def test_top_k_bad_env_falls_back_to_default():
    os.environ[ENV_TOP_K] = "not_an_int"
    assert _top_k() == mem0_litellm_hook.DEFAULT_TOP_K


def test_infer_flag_default_true():
    assert _infer_flag() is True


def test_infer_flag_false_when_env_is_zero():
    os.environ[ENV_INFER] = "0"
    assert _infer_flag() is False


def test_infer_flag_true_when_env_is_one():
    os.environ[ENV_INFER] = "1"
    assert _infer_flag() is True


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def test_build_mem0_config_empty_when_no_env():
    assert _build_mem0_config() == {}


def test_build_mem0_config_forwards_llm():
    os.environ[ENV_LLM_PROVIDER] = "openai"
    os.environ[ENV_LLM_MODEL] = "gpt-4o-mini"
    os.environ[ENV_LLM_BASE_URL] = "https://api.openai.com/v1"
    os.environ["OPENAI_API_KEY"] = "sk-test-xxx"
    cfg = _build_mem0_config()
    assert cfg["llm"]["provider"] == "openai"
    assert cfg["llm"]["config"]["model"] == "gpt-4o-mini"
    assert cfg["llm"]["config"]["openai_base_url"] == "https://api.openai.com/v1"
    assert cfg["llm"]["config"]["api_key"] == "sk-test-xxx"


def test_build_mem0_config_forwards_embedder():
    os.environ[ENV_EMB_PROVIDER] = "openai"
    os.environ[ENV_EMB_MODEL] = "text-embedding-3-small"
    cfg = _build_mem0_config()
    assert cfg["embedder"]["provider"] == "openai"
    assert cfg["embedder"]["config"]["model"] == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Content flattening + extraction helpers
# ---------------------------------------------------------------------------


def test_flatten_content_string():
    assert _flatten_content("hello") == "hello"


def test_flatten_content_multimodal_list():
    parts = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
        {"type": "image_url", "image_url": "ignored"},
    ]
    assert _flatten_content(parts) == "first\nsecond"


def test_flatten_content_other_returns_empty():
    assert _flatten_content(None) == ""
    assert _flatten_content(42) == ""


def test_extract_response_text_plain():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hi there", tool_calls=None)
            )
        ]
    )
    assert _extract_response_text(response) == "hi there"


def test_extract_response_text_tool_calls():
    fn = SimpleNamespace(name="bash", arguments='{"cmd": "ls"}')
    tc = SimpleNamespace(id="call_1", function=fn)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))]
    )
    payload = _extract_response_text(response)
    decoded = json.loads(payload)
    assert decoded[0]["function"]["name"] == "bash"


def test_extract_response_text_empty_on_no_message():
    response = SimpleNamespace(choices=[SimpleNamespace(message=None)])
    assert _extract_response_text(response) == ""


# ---------------------------------------------------------------------------
# Recall query + memory-block formatting
# ---------------------------------------------------------------------------


def test_recall_query_prefers_assistant():
    msgs = [
        {"role": "user", "content": "the issue text"},
        {"role": "assistant", "content": "the most recent agent thinking"},
    ]
    assert _recall_query(msgs) == "the most recent agent thinking"


def test_recall_query_falls_back_to_user_on_step_0():
    msgs = [{"role": "user", "content": "step 0 issue text"}]
    assert _recall_query(msgs) == "step 0 issue text"


def test_recall_query_skips_whitespace_assistant():
    msgs = [
        {"role": "user", "content": "the real query"},
        {"role": "assistant", "content": "   \n  "},  # whitespace-only
    ]
    assert _recall_query(msgs) == "the real query"


def test_format_memories_block_with_source_ids():
    hits = [
        {"memory": "first chunk", "metadata": {"source_id": "step:0"}},
        {"memory": "second chunk", "metadata": {"source_id": "step:1"}},
    ]
    block = _format_memories_block(hits)
    assert "<recalled_episodes>" in block
    assert "[step:0] first chunk" in block
    assert "[step:1] second chunk" in block
    assert "</recalled_episodes>" in block


def test_format_memories_block_empty_returns_empty():
    assert _format_memories_block([]) == ""


def test_prepend_to_last_user_mutates_in_place():
    msgs = [{"role": "user", "content": "the actual question"}]
    ok = _prepend_to_last_user(msgs, "<recalled>x</recalled>")
    assert ok is True
    assert msgs[0]["content"].startswith("<recalled>x</recalled>")
    assert "the actual question" in msgs[0]["content"]


def test_prepend_to_last_user_no_user_returns_false():
    msgs = [{"role": "system", "content": "system only"}]
    assert _prepend_to_last_user(msgs, "block") is False


# ---------------------------------------------------------------------------
# CustomLogger end-to-end with stubbed mem0.Memory
# ---------------------------------------------------------------------------


def _install_fake_mem0_module(monkeypatch, fake_memory_class):
    """Inject a fake mem0 module into sys.modules + reset the singleton."""
    mem0_litellm_hook._memory_instance = None
    fake_mem0 = SimpleNamespace(Memory=fake_memory_class)
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0)


def test_log_pre_api_call_runs_search_and_prepends(monkeypatch, tmp_path):
    os.environ[ENV_USER_ID] = "task-1"
    os.environ[ENV_METRICS_FILE] = str(tmp_path / "metrics.jsonl")

    class FakeMemory:
        def __init__(self):
            self.calls = []

        def search(self, query, filters, top_k):
            self.calls.append((query, filters, top_k))
            return {
                "results": [
                    {"memory": "echo hello", "metadata": {"source_id": "step:0"}}
                ]
            }

        @classmethod
        def from_config(cls, config_dict):
            return cls()

    _install_fake_mem0_module(monkeypatch, FakeMemory)

    # Skip litellm by registering only the injector class manually
    import litellm  # type: ignore  # noqa: F401  -- pytest will fail loud if missing

    injector_cls = mem0_litellm_hook._build_logger_class()
    injector = injector_cls()
    msgs = [{"role": "user", "content": "fix the bug"}]
    injector.log_pre_api_call(model="gpt-4o", messages=msgs, kwargs={})
    assert msgs[0]["content"].startswith("<recalled_episodes>")
    assert "step:0" in msgs[0]["content"]

    # Metrics file should have a recall record
    metrics_lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(metrics_lines) == 1
    recall_record = json.loads(metrics_lines[0])
    assert recall_record["event"] == "recall"
    assert recall_record["memories"] == 1


def test_log_success_event_calls_add_with_infer_false(monkeypatch, tmp_path):
    os.environ[ENV_USER_ID] = "task-1"
    os.environ[ENV_INFER] = "0"
    os.environ[ENV_METRICS_FILE] = str(tmp_path / "metrics.jsonl")

    class FakeMemory:
        def __init__(self):
            self.added: list = []

        def search(self, query, filters, top_k):
            return {"results": []}

        def add(self, messages, user_id, metadata, infer):
            self.added.append({
                "messages": messages, "user_id": user_id,
                "metadata": metadata, "infer": infer,
            })
            return {}

        @classmethod
        def from_config(cls, config_dict):
            return cls()

    _install_fake_mem0_module(monkeypatch, FakeMemory)
    injector_cls = mem0_litellm_hook._build_logger_class()
    injector = injector_cls()

    msgs = [{"role": "user", "content": "fix the bug"}]
    # Prime the recall counter
    injector.log_pre_api_call(model="gpt-4o", messages=msgs, kwargs={})

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="ls && cat file.py", tool_calls=None
        ))]
    )
    injector.log_success_event(
        kwargs={"messages": msgs},
        response_obj=response,
        start_time=0.0,
        end_time=0.1,
    )
    mem_instance = mem0_litellm_hook._memory_instance
    assert len(mem_instance.added) == 1
    added = mem_instance.added[0]
    assert added["infer"] is False  # mem0_no_infer flag honoured
    assert added["user_id"] == "task-1"
    assert added["metadata"]["source_id"].startswith("step:")
    assert "Action:" in added["messages"][0]["content"]
    assert "Observation:" in added["messages"][0]["content"]


def test_first_step_only_skips_subsequent_recalls(monkeypatch, tmp_path):
    os.environ[ENV_USER_ID] = "task-1"
    os.environ[ENV_FIRST_STEP_ONLY] = "1"
    os.environ[ENV_METRICS_FILE] = str(tmp_path / "metrics.jsonl")

    search_count = [0]

    class FakeMemory:
        def search(self, query, filters, top_k):
            search_count[0] += 1
            return {"results": []}

        @classmethod
        def from_config(cls, config_dict):
            return cls()

    _install_fake_mem0_module(monkeypatch, FakeMemory)
    injector_cls = mem0_litellm_hook._build_logger_class()
    injector = injector_cls()

    msgs = [{"role": "user", "content": "issue text"}]
    injector.log_pre_api_call(model="gpt-4o", messages=msgs, kwargs={})
    # subsequent call should be gated
    msgs2 = [
        {"role": "user", "content": "issue text"},
        {"role": "assistant", "content": "step 1 thinking"},
    ]
    injector.log_pre_api_call(model="gpt-4o", messages=msgs2, kwargs={})
    injector.log_pre_api_call(model="gpt-4o", messages=msgs2, kwargs={})
    assert search_count[0] == 1  # only step 0 hit


def test_disable_store_skips_add(monkeypatch, tmp_path):
    os.environ[ENV_USER_ID] = "task-1"
    os.environ[ENV_DISABLE_STORE] = "1"
    os.environ[ENV_METRICS_FILE] = str(tmp_path / "metrics.jsonl")

    add_count = [0]

    class FakeMemory:
        def search(self, query, filters, top_k):
            return {"results": []}

        def add(self, messages, user_id, metadata, infer):
            add_count[0] += 1
            return {}

        @classmethod
        def from_config(cls, config_dict):
            return cls()

    _install_fake_mem0_module(monkeypatch, FakeMemory)
    injector_cls = mem0_litellm_hook._build_logger_class()
    injector = injector_cls()

    msgs = [{"role": "user", "content": "issue text"}]
    injector.log_pre_api_call(model="gpt-4o", messages=msgs, kwargs={})
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="cmd", tool_calls=None
        ))]
    )
    injector.log_success_event(
        kwargs={"messages": msgs},
        response_obj=response,
        start_time=0.0,
        end_time=0.1,
    )
    assert add_count[0] == 0


# ---------------------------------------------------------------------------
# register_mem0_hook contract
# ---------------------------------------------------------------------------


def test_register_returns_false_when_user_id_unset():
    # No env → silent no-op, no exception
    assert mem0_litellm_hook.register_mem0_hook() is False


def test_register_idempotent_within_subprocess(monkeypatch):
    os.environ[ENV_USER_ID] = "task-1"

    import litellm  # type: ignore

    fake_callbacks: list = []
    monkeypatch.setattr(litellm, "callbacks", fake_callbacks)

    class FakeMemory:
        @classmethod
        def from_config(cls, config_dict):
            return cls()

    _install_fake_mem0_module(monkeypatch, FakeMemory)
    assert mem0_litellm_hook.register_mem0_hook() is True
    n_after_first = len(litellm.callbacks)
    assert n_after_first == 1
    # Re-register: should not add a second copy
    assert mem0_litellm_hook.register_mem0_hook() is True
    assert len(litellm.callbacks) == n_after_first


def test_sitecustomize_stub_mem0_constant_shape():
    """Locks the registration block emitted into sitecustomize.py.

    The stub must reference both ``register_mem0_hook`` AND tolerate
    the two import paths (local sibling vs package-installed).
    """
    assert "register_mem0_hook" in SITECUSTOMIZE_STUB_MEM0
    assert "import mem0_litellm_hook" in SITECUSTOMIZE_STUB_MEM0
    assert (
        "from cross_bench.execution.scaffolds.mem0_litellm_hook"
        in SITECUSTOMIZE_STUB_MEM0
    )
    assert "_status_mem0(" in SITECUSTOMIZE_STUB_MEM0


# ---------------------------------------------------------------------------
# seal._resolve_mem0_hook_config — decision tree
# ---------------------------------------------------------------------------


def _make_mem0_middleware_mock(user_id="task-1", retrieval_limit=20, infer=False, params=None):
    """Build a SimpleNamespace shaped like Mem0V2EmbeddedMiddleware."""
    config = SimpleNamespace(params=params or {})
    return SimpleNamespace(
        _memory=object(),  # a real mem0.Memory instance in prod
        _user_id=user_id,
        _retrieval_limit=retrieval_limit,
        _infer=infer,
        _config=config,
    )


def test_resolve_mem0_returns_none_for_no_memory():
    assert _resolve_mem0_hook_config(None) is None


def test_resolve_mem0_returns_none_for_unrelated_object():
    other = SimpleNamespace(_some_other_attr="x")
    assert _resolve_mem0_hook_config(other) is None


def test_resolve_mem0_returns_env_for_mem0_middleware():
    mem = _make_mem0_middleware_mock(user_id="t1", retrieval_limit=15, infer=False)
    env = _resolve_mem0_hook_config(mem)
    assert env is not None
    assert env[ENV_USER_ID] == "t1"
    assert env[ENV_TOP_K] == "15"
    assert env[ENV_INFER] == "0"


def test_resolve_mem0_forwards_inherited_llm_config():
    mem = _make_mem0_middleware_mock(
        params={
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_base_url": "https://api.openai.com/v1",
            "embedder_provider": "openai",
            "embedder_model": "text-embedding-3-small",
        }
    )
    env = _resolve_mem0_hook_config(mem)
    assert env[ENV_LLM_PROVIDER] == "openai"
    assert env[ENV_LLM_MODEL] == "gpt-4o-mini"
    assert env[ENV_LLM_BASE_URL] == "https://api.openai.com/v1"
    assert env[ENV_EMB_PROVIDER] == "openai"
    assert env[ENV_EMB_MODEL] == "text-embedding-3-small"


def test_resolve_mem0_first_step_only_sets_gating_envs():
    mem = _make_mem0_middleware_mock()
    env = _resolve_mem0_hook_config(
        mem, benchmark_params={"memory_first_step_only": True}
    )
    assert env[ENV_FIRST_STEP_ONLY] == "1"
    assert env[ENV_DISABLE_STORE] == "1"


def test_resolve_mem0_retrieve_only_sets_disable_store():
    mem = _make_mem0_middleware_mock()
    env = _resolve_mem0_hook_config(
        mem, benchmark_params={"memory_retrieve_only": True}
    )
    assert ENV_FIRST_STEP_ONLY not in env  # only disable_store fires
    assert env[ENV_DISABLE_STORE] == "1"


def test_resolve_mem0_default_neither_gating_env():
    mem = _make_mem0_middleware_mock()
    env = _resolve_mem0_hook_config(mem, benchmark_params={})
    assert ENV_FIRST_STEP_ONLY not in env
    assert ENV_DISABLE_STORE not in env


def test_resolve_mem0_empty_user_id_returns_none():
    mem = _make_mem0_middleware_mock(user_id="")
    assert _resolve_mem0_hook_config(mem) is None


# ---------------------------------------------------------------------------
# _aggregate_mem0_metrics
# ---------------------------------------------------------------------------


def test_aggregate_mem0_metrics_missing_file_returns_empty(tmp_path):
    assert _aggregate_mem0_metrics(tmp_path / "missing.jsonl") == {}


def test_aggregate_mem0_metrics_sums_recall_and_store(tmp_path):
    path = tmp_path / "mem0_metrics.jsonl"
    records = [
        {"event": "recall", "call_index": 0, "tokens": 100, "latency_ms": 50.0, "memories": 5},
        {"event": "recall", "call_index": 1, "tokens": 80, "latency_ms": 40.0, "memories": 3},
        {"event": "store", "call_index": 0, "tokens_estimate": 200, "latency_ms": 30.0},
        {"event": "store", "call_index": 1, "tokens_estimate": 150, "latency_ms": 25.0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records))
    totals = _aggregate_mem0_metrics(path)
    assert totals["mem0_recall_calls"] == 2
    assert totals["mem0_recall_tokens"] == 180
    assert totals["mem0_recall_latency_ms"] == 90.0
    assert totals["mem0_recall_memories"] == 8
    assert totals["mem0_store_calls"] == 2
    assert totals["mem0_store_tokens_estimate"] == 350


def test_aggregate_mem0_metrics_skips_malformed_lines(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"event": "recall", "tokens": 50, "memories": 1, "latency_ms": 5.0}\n'
        "not_json\n"
        ""  # blank line
        '\n{"event": "store", "tokens_estimate": 100, "latency_ms": 5.0}\n'
    )
    totals = _aggregate_mem0_metrics(path)
    assert totals["mem0_recall_calls"] == 1
    assert totals["mem0_store_calls"] == 1


# ---------------------------------------------------------------------------
# Subprocess env wiring (mem0_env overlay)
# ---------------------------------------------------------------------------


def test_render_episode_byte_equal_to_mini_swe_agent():
    """LOAD-BEARING: mem0 episode shape must match S18 byte-for-byte.

    Both ``_render_episode`` implementations are called on the same
    ``(cmd, obs)`` pair; the test asserts byte equality. Drift on
    either side fails CI immediately. Locks S25 §3 / TS sec.3 — the
    SEAL mem0 cell and the mini-SWE-agent mem0 cell ingest the same
    episode shape so cross-scaffold comparison stays honest.
    """
    from cross_bench.execution.scaffolds.mem0_litellm_hook import (
        _render_episode as _seal_render,
    )
    from cross_bench.execution.scaffolds.mini_swe_agent import (
        _render_episode as _mini_render,
    )
    cmd = "cat foo.py"
    obs = "<output>FILE_CONTENT</output>"
    assert _seal_render(cmd, obs) == _mini_render(cmd, obs)
    # Explicit shape lock (in case both sides drift identically and
    # the cross-check passes vacuously).
    assert _seal_render(cmd, obs) == (
        "Action:\n```\ncat foo.py\n```\n\n"
        "Observation:\n<output>FILE_CONTENT</output>"
    )


def test_sitecustomize_stub_mem0_execs_without_nameerror(tmp_path, monkeypatch):
    """LOAD-BEARING: the appended mem0 stanza must not NameError on
    ``_status`` when the host sitecustomize.py was written by the
    step-counter writer (which doesn't define ``_status``).

    Reviewer-bot caught: SITECUSTOMIZE_STUB_MEM0 originally called
    ``_status(...)`` six times, but ``step_counter_hook.SITECUSTOMIZE_STUB``
    doesn't define ``_status``. When mem0 is active, seal.py appends
    the mem0 block onto the step-counter-only stub →
    NameError on every status call → mem0_status.txt never written.

    The fix is a defensive ``_status_mem0`` definition at the top of
    SITECUSTOMIZE_STUB_MEM0; this test exercises that path.
    """
    status_path = tmp_path / "mem0_status.txt"
    monkeypatch.setenv("CROSSBENCH_MEM0_STATUS_FILE", str(status_path))
    # Don't set USER_ID — register_mem0_hook returns False silently,
    # but the _status_mem0 path must still write its log line.
    monkeypatch.delenv(ENV_USER_ID, raising=False)
    # Execute the stub against an empty-ish namespace (mimicking
    # site-init when only the step-counter stub preceded us — no
    # ``_status`` defined).
    namespace: dict = {}
    # NameError would fire here if _status_mem0 wasn't defined defensively.
    exec(mem0_litellm_hook.SITECUSTOMIZE_STUB_MEM0, namespace)
    # The status file should contain the "mem0_disabled" line.
    contents = status_path.read_text()
    assert "mem0_disabled" in contents


def test_subprocess_env_carries_mem0_overlay(tmp_path):
    """Smoke: _build_subprocess_env + manual overlay produces the expected dict."""
    fake_root = tmp_path / "swe_agent"
    fake_root.mkdir()
    env = _build_subprocess_env(fake_root)
    # Manual overlay (execute_seal does this when mem0_env is not None)
    env.update({
        ENV_USER_ID: "task-99",
        ENV_TOP_K: "10",
        ENV_INFER: "0",
    })
    assert env[ENV_USER_ID] == "task-99"
    assert env[ENV_TOP_K] == "10"
    assert env[ENV_INFER] == "0"
    # PYTHONPATH must still have the swe_agent root
    assert str(fake_root) in env["PYTHONPATH"]
