"""Test execution engine for live testing."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import discord
    from discord.ext import commands

logger = logging.getLogger("chorus.testing.runner")

# Default timeout per test in seconds
DEFAULT_TEST_TIMEOUT = 30.0


@dataclass
class TestResult:
    """Result of a single test execution."""

    __test__ = False  # Prevent pytest collection

    name: str
    passed: bool
    duration_ms: float
    detail: str | None = None
    error: str | None = None


@dataclass
class SuiteResult:
    """Aggregated result of running a test suite."""

    suite_name: str
    results: list[TestResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


class TestRunner:
    """Executes live test functions and collects results."""

    __test__ = False  # Prevent pytest collection

    def __init__(self, bot: commands.Bot, channel: discord.TextChannel) -> None:
        self.bot = bot
        self.channel = channel
        self.timeout: float = DEFAULT_TEST_TIMEOUT

    async def run_suite(self, suite_name: str, tests: list[Callable[..., Any]]) -> SuiteResult:
        """Run a named suite of test functions."""
        start = time.monotonic()
        results: list[TestResult] = []

        for test_fn in tests:
            result = await self.run_test(test_fn)
            results.append(result)

        total_ms = (time.monotonic() - start) * 1000
        return SuiteResult(suite_name=suite_name, results=results, total_ms=total_ms)

    async def run_test(self, test_fn: Callable[..., Any]) -> TestResult:
        """Run a single test function, timing it and catching exceptions."""
        name = test_fn.__name__
        start = time.monotonic()

        try:
            detail = await asyncio.wait_for(test_fn(self), timeout=self.timeout)
            duration_ms = (time.monotonic() - start) * 1000
            return TestResult(
                name=name,
                passed=True,
                duration_ms=duration_ms,
                detail=detail,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration_ms,
                error=f"Timed out after {self.timeout}s",
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            tb = traceback.format_exception_only(type(exc), exc)
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration_ms,
                error="".join(tb).strip(),
            )
