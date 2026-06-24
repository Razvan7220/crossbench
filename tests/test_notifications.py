"""Unit tests for the per-task Slack notification module.

These tests intentionally lock only the *stable* behaviours (env-var gate,
exception swallowing, helper formatters). The message *shape* is still
iterating and is verified by sending real Slack messages during dev; we
don't lock its layout in tests to avoid churn.
"""

from __future__ import annotations

from unittest import mock

import pytest

from cross_bench import notifications
from cross_bench.notifications import (
    _fmt_duration,
    _fmt_pct,
    _fmt_tokens,
    notify_task_done,
)


class _FakeStatus:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeResult:
    def __init__(
        self,
        task_id: str = "instance_repo__task-abc123",
        status: str = "pass",
        error: str = "",
        evaluation: dict | None = None,
    ) -> None:
        self.task_id = task_id
        self.status = _FakeStatus(status)
        self.error = error
        self.evaluation = evaluation or {}


class _FakeTask:
    def __init__(self, task_id: str = "instance_repo__task-abc123") -> None:
        self.task_id = task_id
        self.metadata = {"repo": "test/repo", "dataset_offset": 1, "dataset_total": 731}


@pytest.fixture(autouse=True)
def _reset_ecs_metadata_cache():
    notifications._ecs_meta_cache = None
    yield
    notifications._ecs_meta_cache = None


# ── Formatter unit tests ────────────────────────────────────────────────────


def test_fmt_pct_renders_fraction_as_integer_percent():
    assert _fmt_pct(0.0) == "0%"
    assert _fmt_pct(0.5) == "50%"
    assert _fmt_pct(1.0) == "100%"


def test_fmt_pct_handles_none_and_garbage():
    assert _fmt_pct(None) == "—"
    assert _fmt_pct("not-a-number") == "—"


def test_fmt_duration_picks_units():
    assert _fmt_duration(0) == "—"
    assert _fmt_duration(45) == "45s"
    assert _fmt_duration(125) == "2.1m"
    assert _fmt_duration(7200) == "2.0h"


def test_fmt_tokens_scales():
    assert _fmt_tokens(None) == "—"
    assert _fmt_tokens(42) == "42"
    assert _fmt_tokens(1500) == "1.5k"
    assert _fmt_tokens(2_500_000) == "2.5M"


# ── Gate behaviour ──────────────────────────────────────────────────────────


def test_default_gate_off_is_noop_for_done(monkeypatch):
    monkeypatch.delenv("CROSSBENCH_SLACK_PER_TASK", raising=False)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.invalid/")
    with mock.patch("cross_bench.notifications.urllib.request.urlopen") as urlopen:
        notify_task_done(_FakeResult(), "run-xyz")
    urlopen.assert_not_called()


def test_gate_on_but_no_webhook_is_noop(monkeypatch):
    monkeypatch.setenv("CROSSBENCH_SLACK_PER_TASK", "1")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    with mock.patch("cross_bench.notifications.urllib.request.urlopen") as urlopen:
        notify_task_done(_FakeResult(), "run-xyz")
    urlopen.assert_not_called()


def test_webhook_failure_is_swallowed_for_done(monkeypatch):
    monkeypatch.setenv("CROSSBENCH_SLACK_PER_TASK", "1")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.invalid/")
    with mock.patch(
        "cross_bench.notifications.urllib.request.urlopen",
        side_effect=RuntimeError("network down"),
    ):
        notify_task_done(_FakeResult(), "run-xyz")


def test_done_posts_when_gate_on_and_webhook_set(monkeypatch):
    monkeypatch.setenv("CROSSBENCH_SLACK_PER_TASK", "1")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.invalid/")
    captured: dict = {}

    def _fake_urlopen(req, *args, **kwargs):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["method"] = req.get_method()
        return mock.MagicMock(
            __enter__=lambda s: s, __exit__=lambda *a: None, read=lambda: b""
        )

    with mock.patch("cross_bench.notifications.urllib.request.urlopen", side_effect=_fake_urlopen):
        notify_task_done(_FakeResult(evaluation={"resolved": True}), "run-xyz")

    assert captured["url"] == "https://example.invalid/"
    assert captured["method"] == "POST"
    assert captured["body"]  # non-empty payload
