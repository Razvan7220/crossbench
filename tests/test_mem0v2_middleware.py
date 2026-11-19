"""Smoke tests for Mem0V2EmbeddedMiddleware error handling (v2.0.0 compatible).

These tests verify that:
1. Empty prompts don't produce empty-content messages
2. Exceptions in add() / search() are caught and don't crash the pipeline
3. search() uses filters={"user_id": ...} (v2.0.0 API)
4. add() uses top-level user_id kwarg
5. Speaker/timestamp metadata is included in messages
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Stub the mem0 package so we can import the middleware without mem0 installed
_mem0_stub = ModuleType("mem0")
_mem0_stub.Memory = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("mem0", _mem0_stub)

from cross_bench.models import MemoryConfig  # noqa: E402
from cross_bench.systems.memory.mem0v2_middleware import Mem0V2EmbeddedMiddleware  # noqa: E402


@pytest.fixture
def mock_memory():
    """A MagicMock that stands in for mem0.Memory."""
    m = MagicMock()
    m.search.return_value = {"results": []}
    m.add.return_value = {"results": []}
    m.delete_all.return_value = None
    return m


@pytest.fixture
def middleware(mock_memory):
    """Construct Mem0V2EmbeddedMiddleware with a mocked Memory backend."""
    mw = Mem0V2EmbeddedMiddleware.__new__(Mem0V2EmbeddedMiddleware)
    mw._config = MemoryConfig(
        provider="mem0v2_embedded", enabled=True, params={"user_id": "test_user"}
    )
    mw._user_id = "test_user"
    mw._agent_id = None
    mw._retrieval_limit = 20
    # Mirror the upstream-default that ``__init__`` would set; individual
    # ``infer=False`` test cases override this attribute directly.
    mw._infer = True
    mw._memory = mock_memory
    return mw


class TestPostResponse:
    """Tests for post_response message construction and error handling."""

    def test_empty_prompt_sends_single_user_message(self, middleware, mock_memory):
        """When prompt is empty (conversation ingestion), only one user message should be sent."""
        asyncio.run(middleware.post_response("", "Jon: Hello there", {}))

        mock_memory.add.assert_called_once()
        call_kwargs = mock_memory.add.call_args[1]
        messages = call_kwargs["messages"]

        # Should be a single user message, NOT an empty user + assistant pair
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Jon: Hello there"

    def test_whitespace_prompt_sends_single_user_message(self, middleware, mock_memory):
        """Whitespace-only prompt should also be treated as empty."""
        asyncio.run(middleware.post_response("   ", "Some text", {}))

        call_kwargs = mock_memory.add.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "Some text"

    def test_nonempty_prompt_sends_user_assistant_pair(self, middleware, mock_memory):
        """Normal Q&A flow sends both user and assistant messages."""
        asyncio.run(middleware.post_response("What is X?", "X is Y.", {}))

        call_kwargs = mock_memory.add.call_args[1]
        messages = call_kwargs["messages"]

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "What is X?"}
        assert messages[1] == {"role": "assistant", "content": "X is Y."}

    def test_add_exception_is_caught(self, middleware, mock_memory):
        """mem0 add() raising should NOT propagate — just log a warning."""
        mock_memory.add.side_effect = Exception("OpenAI 400 BadRequest")

        # Should not raise
        asyncio.run(middleware.post_response("", "Some text", {}))

    def test_add_openai_400_is_caught(self, middleware, mock_memory):
        """Simulate the exact OpenAI BadRequestError scenario."""
        mock_memory.add.side_effect = RuntimeError(
            "Error code: 400 - We could not parse the JSON body"
        )

        # Should not raise
        asyncio.run(middleware.post_response("prompt", "response", {}))

    def test_long_content_is_truncated(self, middleware, mock_memory):
        """Messages longer than 8000 chars should be truncated."""
        long_text = "x" * 10000
        asyncio.run(middleware.post_response("", long_text, {}))

        call_kwargs = mock_memory.add.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages[0]["content"]) == 8000

    def test_add_uses_toplevel_user_id(self, middleware, mock_memory):
        """add() should pass user_id as a top-level kwarg (v2.0.0 API)."""
        asyncio.run(middleware.post_response("", "Hello", {}))

        call_kwargs = mock_memory.add.call_args[1]
        assert call_kwargs["user_id"] == "test_user"
        # Should NOT have a filters key — add() uses top-level user_id
        assert "filters" not in call_kwargs

    def test_speaker_timestamp_in_content(self, middleware, mock_memory):
        """Speaker and timestamp should be prepended to message content."""
        asyncio.run(middleware.post_response("", "Hello world", {
            "speaker_id": "Amy",
            "timestamp": "2023-05-08 13:56:00",
        }))

        call_kwargs = mock_memory.add.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert "[2023-05-08 13:56:00 | Amy]" in content
        assert "Hello world" in content

    def test_speaker_timestamp_metadata(self, middleware, mock_memory):
        """Speaker and timestamp should be passed as metadata."""
        asyncio.run(middleware.post_response("", "Hello", {
            "speaker_id": "Bob",
            "timestamp": "2023-06-01 10:00:00",
        }))

        call_kwargs = mock_memory.add.call_args[1]
        assert call_kwargs["metadata"]["speaker_id"] == "Bob"
        assert call_kwargs["metadata"]["timestamp"] == "2023-06-01 10:00:00"


class TestPrePrompt:
    """Tests for pre_prompt search and error handling."""

    def test_search_exception_returns_original_prompt(self, middleware, mock_memory):
        """mem0 search() raising should return the original prompt unchanged."""
        mock_memory.search.side_effect = Exception("Connection error")

        result = asyncio.run(middleware.pre_prompt("What happened?", {}))
        assert result == "What happened?"

    def test_search_uses_filters_not_toplevel_user_id(self, middleware, mock_memory):
        """search() must use filters={"user_id": ...}, NOT top-level user_id (v2.0.0)."""
        mock_memory.search.return_value = {"results": []}

        asyncio.run(middleware.pre_prompt("Hello", {}))

        call_kwargs = mock_memory.search.call_args[1]
        # v2.0.0: user_id must be in filters
        assert "filters" in call_kwargs
        assert call_kwargs["filters"]["user_id"] == "test_user"
        # Must NOT have top-level user_id
        assert "user_id" not in call_kwargs

    def test_search_returns_memories(self, middleware, mock_memory):
        """When search returns memories, they should be prepended."""
        mock_memory.search.return_value = {
            "results": [
                {"memory": "Jon has a dance studio"},
                {"memory": "Jon likes natural light"},
            ]
        }

        result = asyncio.run(middleware.pre_prompt("Tell me about Jon", {}))
        assert "Jon has a dance studio" in result
        assert "Jon likes natural light" in result
        assert "Tell me about Jon" in result

    def test_search_returns_memories_with_timestamps(self, middleware, mock_memory):
        """Memories with created_at should include timestamps in output."""
        mock_memory.search.return_value = {
            "results": [
                {"memory": "Amy likes hiking", "created_at": "2023-05-08T14:00:00Z"},
            ]
        }

        result = asyncio.run(middleware.pre_prompt("Tell me about Amy", {}))
        assert "[2023-05-08T14:00:00Z]" in result
        assert "Amy likes hiking" in result

    def test_metadata_timestamp_preferred_over_created_at(
        self, middleware, mock_memory
    ):
        """metadata.timestamp (conversation time) should win over created_at (ingestion time)."""
        mock_memory.search.return_value = {
            "results": [
                {
                    "memory": "Amy likes hiking",
                    # mem0 always overrides this to ingestion-now
                    "created_at": "2026-04-22T11:24:32+00:00",
                    "metadata": {"timestamp": "2023-05-08 13:56:00"},
                },
            ]
        }

        result = asyncio.run(middleware.pre_prompt("Tell me about Amy", {}))
        assert "[2023-05-08 13:56:00]" in result
        assert "2026-04-22" not in result

    def test_empty_search_returns_original(self, middleware, mock_memory):
        """Empty search results should return the original prompt."""
        mock_memory.search.return_value = {"results": []}

        result = asyncio.run(middleware.pre_prompt("Hello", {}))
        assert result == "Hello"

    def test_search_passes_top_k(self, middleware, mock_memory):
        """search() should pass top_k from retrieval_limit config."""
        mock_memory.search.return_value = {"results": []}

        asyncio.run(middleware.pre_prompt("Hello", {}))

        call_kwargs = mock_memory.search.call_args[1]
        assert call_kwargs["top_k"] == 20


# ---------------------------------------------------------------------------
# `infer` param plumbing (the ``mem0_no_infer`` mixin's contract)
# ---------------------------------------------------------------------------
#
# Lets a profile turn off mem0's `infer=True` LLM-fact-extraction step so
# ingested chunks are stored verbatim instead of summarised into NL facts.
# Required for code-RAG benchmarks (CrossCodeEval, RepoBench) where
# identifier exact-match is graded — see `configs/profiles/mem0_no_infer.toml`
# and deviations.md M9.


class TestInferParamPlumbing:
    """The ``infer`` profile param must propagate from MemoryConfig to
    every ``memory.add(...)`` call. Default is True (mem0 upstream
    default); ``mem0_no_infer`` mixin flips it to False."""

    def test_default_infer_is_true(self):
        """Without an ``infer`` key in params, the middleware must
        default to True to match mem0's upstream ``Memory.add`` default
        and preserve historical conversational-benchmark behavior."""
        # Bypass __init__'s real Memory construction by carrying just
        # enough state in via params — the attribute we care about is
        # `_infer`, which __init__ reads from params.
        mw = Mem0V2EmbeddedMiddleware.__new__(Mem0V2EmbeddedMiddleware)
        mw._config = MemoryConfig(
            provider="mem0v2_embedded", enabled=True, params={}
        )
        # Re-run the relevant line of __init__ (don't construct mem0):
        mw._infer = bool(mw._config.params.get("infer", True))
        assert mw._infer is True

    def test_params_infer_false_sets_attribute_false(self):
        """``params.infer = false`` (from the mem0_no_infer mixin)
        must land on ``self._infer = False``."""
        mw = Mem0V2EmbeddedMiddleware.__new__(Mem0V2EmbeddedMiddleware)
        mw._config = MemoryConfig(
            provider="mem0v2_embedded", enabled=True, params={"infer": False}
        )
        mw._infer = bool(mw._config.params.get("infer", True))
        assert mw._infer is False

    def test_post_response_forwards_infer_true(self, middleware, mock_memory):
        """Default fixture has ``_infer = True``; the call must surface that."""
        asyncio.run(middleware.post_response("q", "a", {}))
        assert mock_memory.add.call_args[1]["infer"] is True

    def test_post_response_forwards_infer_false(self, middleware, mock_memory):
        """Flipping ``_infer`` on the instance MUST flow through to
        the mem0 call site — that's the whole point of the mixin."""
        middleware._infer = False
        asyncio.run(middleware.post_response("q", "a", {}))
        assert mock_memory.add.call_args[1]["infer"] is False

    def test_post_response_forwards_infer_alongside_other_kwargs(
        self, middleware, mock_memory
    ):
        """``infer`` adds to the existing kwargs (messages, user_id,
        metadata) — it does NOT replace or get swallowed."""
        asyncio.run(middleware.post_response("q", "a", {"timestamp": "2023"}))
        kwargs = mock_memory.add.call_args[1]
        assert "infer" in kwargs
        assert kwargs["user_id"] == "test_user"
        assert "messages" in kwargs


# ---------------------------------------------------------------------------
# `infer` plumbing on the API variant
# ---------------------------------------------------------------------------


class TestApiInferParamPlumbing:
    """Same contract for the cloud ``Mem0V2ApiMiddleware``."""

    def test_default_infer_is_true_api(self):
        """API variant default also True (matches ``MemoryClient.add``
        upstream default)."""
        from cross_bench.systems.memory.mem0v2_middleware import Mem0V2ApiMiddleware

        mw = Mem0V2ApiMiddleware.__new__(Mem0V2ApiMiddleware)
        mw._config = MemoryConfig(
            provider="mem0v2_api", enabled=True, params={}
        )
        mw._infer = bool(mw._config.params.get("infer", True))
        assert mw._infer is True

    def test_params_infer_false_api(self):
        from cross_bench.systems.memory.mem0v2_middleware import Mem0V2ApiMiddleware

        mw = Mem0V2ApiMiddleware.__new__(Mem0V2ApiMiddleware)
        mw._config = MemoryConfig(
            provider="mem0v2_api", enabled=True, params={"infer": False}
        )
        mw._infer = bool(mw._config.params.get("infer", True))
        assert mw._infer is False

    def test_post_response_forwards_infer_on_api_variant(self):
        """API variant: ``self._client.add(messages, filters=..., infer=...)``."""
        from cross_bench.systems.memory.mem0v2_middleware import Mem0V2ApiMiddleware

        mw = Mem0V2ApiMiddleware.__new__(Mem0V2ApiMiddleware)
        mw._config = MemoryConfig(
            provider="mem0v2_api", enabled=True, params={"user_id": "u"}
        )
        mw._user_id = "u"
        mw._agent_id = None
        mw._app_id = None
        mw._run_id = None
        mw._retrieval_limit = 20
        mw._infer = False  # the mem0_no_infer setting
        mw._client = MagicMock()
        mw._client.add.return_value = {"results": []}

        asyncio.run(mw.post_response("q", "a", {}))

        # mem0v2 API call shape is `client.add(messages, **kwargs)` —
        # `infer` lives in kwargs.
        kwargs = mw._client.add.call_args[1]
        assert kwargs["infer"] is False


class TestBuildEmbeddedConfigAWSBedrock:
    """``_build_embedded_config`` must thread aws_region/aws_profile through.

    Regression lock for the 2026-05-26 incident where every ECS shard of
    a mem0v2 × Bedrock smoke crashed at engine init with
    ``Unsupported LLM provider: anthropic_bedrock``. The fix is two-stage:
    config.py normalizes the provider name to ``aws_bedrock``, and the
    middleware threads the bedrock auth fields into mem0's LLM config.
    """

    def test_aws_bedrock_provider_threads_region(self):
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_provider": "aws_bedrock",
            "llm_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "aws_region": "us-east-1",
        })
        llm = cfg["llm"]
        assert llm["provider"] == "aws_bedrock"
        assert llm["config"]["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert llm["config"]["aws_region"] == "us-east-1"

    def test_aws_bedrock_provider_threads_profile(self):
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_provider": "aws_bedrock",
            "llm_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "aws_profile": "research-dev",
        })
        assert cfg["llm"]["config"]["aws_profile"] == "research-dev"

    def test_aws_bedrock_omits_region_when_unset(self):
        """No aws_region → fall back to boto3 chain. Don't write None."""
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_provider": "aws_bedrock",
            "llm_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        })
        assert "aws_region" not in cfg["llm"]["config"]
        assert "aws_profile" not in cfg["llm"]["config"]

    def test_non_bedrock_provider_ignores_aws_fields(self):
        """``aws_region`` on a non-bedrock provider is silently dropped.

        Defensive: if a future contributor pastes aws_region into an
        openai profile by mistake, the middleware must not pass it on.
        """
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "aws_region": "us-east-1",  # stray
        })
        assert "aws_region" not in cfg["llm"]["config"]


class TestBuildEmbeddedConfigOllama:
    """Ollama embedding plumbing for the 1536-dim GTE profile.

    Mem0's ``OllamaEmbedding`` uses the native ``ollama.Client`` (no
    ``/v1``), while the extractor LLM often rides Ollama's
    OpenAI-compatible endpoint.  The config builder must keep the two
    separate, and Qdrant must be told the embedder dimension instead of
    assuming the 1536 default.
    """

    def test_native_embedder_url_and_dims(self):
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_provider": "openai",
            "llm_model": "qwen2.5:7b-instruct-8t",
            "llm_base_url": "http://localhost:11434/v1",
            "embedder_provider": "ollama",
            "embedder_model": "gte-qwen2:8t",
            "embedder_base_url": "http://localhost:11434",
            "embedder_dims": 1536,
            "vector_store_provider": "qdrant",
            "vector_store_path": "/tmp/qdrant-test",
        })

        assert cfg["embedder"]["config"]["model"] == "gte-qwen2:8t"
        assert cfg["embedder"]["config"]["ollama_base_url"] == "http://localhost:11434"
        assert cfg["embedder"]["config"]["embedding_dims"] == 1536
        assert cfg["vector_store"]["config"]["embedding_model_dims"] == 1536

    def test_embedder_url_falls_back_to_native_llm_url(self):
        """Existing ``ollama_memory`` stacks have no ``embedder_base_url``."""
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_base_url": "http://localhost:11434/v1",
            "embedder_provider": "ollama",
        })

        assert cfg["embedder"]["config"]["ollama_base_url"] == "http://localhost:11434"

    def test_ollama_llm_url_is_native(self):
        """``llm_provider=ollama`` also uses the native client."""
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_provider": "ollama",
            "llm_model": "qwen2.5:7b-instruct-8t",
            "llm_base_url": "http://localhost:11434/v1/",
        })

        assert cfg["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"

    def test_openai_embedder_keeps_v1_suffix(self):
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "embedder_provider": "openai",
            "embedder_model": "text-embedding-3-small",
            "embedder_base_url": "http://localhost:11434/v1",
            "embedder_dims": 768,
        })

        assert cfg["embedder"]["config"]["openai_base_url"] == "http://localhost:11434/v1"
        assert cfg["embedder"]["config"]["embedding_dims"] == 768

    def test_qdrant_dims_not_set_for_other_vector_stores(self):
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "embedder_dims": 1536,
            "vector_store_provider": "chroma",
            "vector_store_path": "/tmp/chroma-test",
        })

        assert "embedding_model_dims" not in cfg["vector_store"]["config"]

    def test_llm_api_key_is_forwarded(self):
        """Profiles set ``llm_api_key``; the OpenAI SDK needs it explicitly."""
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_provider": "openai",
            "llm_base_url": "http://localhost:11434/v1",
            "llm_api_key": "ollama-local",
        })

        assert cfg["llm"]["config"]["api_key"] == "ollama-local"

    def test_ollama_llm_key_is_not_forwarded(self):
        """The native ollama client must not receive an OpenAI-style key."""
        from cross_bench.systems.memory.mem0v2_middleware import _build_embedded_config

        cfg = _build_embedded_config({
            "llm_provider": "ollama",
            "llm_base_url": "http://localhost:11434/v1",
            "llm_api_key": "ollama-local",
        })

        assert "api_key" not in cfg["llm"]["config"]
