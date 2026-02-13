"""Tests for chorus.testing.runner — test execution engine."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from chorus.testing.runner import TestRunner


@pytest.fixture
def runner() -> TestRunner:
    """Create a TestRunner with mock bot and channel."""
    bot = MagicMock()
    bot.latency = 0.042
    channel = MagicMock()
    channel.send = MagicMock()
    r = TestRunner(bot, channel)
    r.timeout = 2.0  # Short timeout for tests
    return r


class TestRunTestPassing:
    @pytest.mark.asyncio
    async def test_run_test_passing(self, runner: TestRunner) -> None:
        async def passing_test(r: TestRunner) -> str | None:
            return "all good"

        result = await runner.run_test(passing_test)
        assert result.passed is True
        assert result.name == "passing_test"
        assert result.detail == "all good"
        assert result.error is None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_test_passing_no_detail(self, runner: TestRunner) -> None:
        async def no_detail_test(r: TestRunner) -> str | None:
            return None

        result = await runner.run_test(no_detail_test)
        assert result.passed is True
        assert result.detail is None


class TestRunTestFailing:
    @pytest.mark.asyncio
    async def test_run_test_failing(self, runner: TestRunner) -> None:
        async def failing_test(r: TestRunner) -> str | None:
            raise RuntimeError("something broke")

        result = await runner.run_test(failing_test)
        assert result.passed is False
        assert result.name == "failing_test"
        assert result.error is not None
        assert "something broke" in result.error
        assert result.detail is None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_test_assertion_error(self, runner: TestRunner) -> None:
        async def assert_test(r: TestRunner) -> str | None:
            raise AssertionError("assertion failed")

        result = await runner.run_test(assert_test)
        assert result.passed is False
        assert "assertion failed" in (result.error or "")


class TestRunTestTimeout:
    @pytest.mark.asyncio
    async def test_run_test_timeout(self, runner: TestRunner) -> None:
        runner.timeout = 0.1

        async def slow_test(r: TestRunner) -> str | None:
            await asyncio.sleep(5)
            return "should not reach"

        result = await runner.run_test(slow_test)
        assert result.passed is False
        assert result.error is not None
        assert "Timed out" in result.error


class TestRunSuite:
    @pytest.mark.asyncio
    async def test_run_suite_all_pass(self, runner: TestRunner) -> None:
        async def t1(r: TestRunner) -> str | None:
            return "ok1"

        async def t2(r: TestRunner) -> str | None:
            return "ok2"

        result = await runner.run_suite("my_suite", [t1, t2])
        assert result.suite_name == "my_suite"
        assert result.passed == 2
        assert result.failed == 0
        assert len(result.results) == 2
        assert result.total_ms >= 0

    @pytest.mark.asyncio
    async def test_run_suite_mixed(self, runner: TestRunner) -> None:
        async def pass_test(r: TestRunner) -> str | None:
            return "ok"

        async def fail_test(r: TestRunner) -> str | None:
            raise ValueError("bad")

        result = await runner.run_suite("mixed", [pass_test, fail_test])
        assert result.passed == 1
        assert result.failed == 1
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_run_suite_empty(self, runner: TestRunner) -> None:
        result = await runner.run_suite("empty", [])
        assert result.passed == 0
        assert result.failed == 0
        assert len(result.results) == 0
