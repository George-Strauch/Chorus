"""Tests for chorus.testing.suites — suite registry validation."""

from __future__ import annotations

import asyncio
import inspect

from chorus.testing.suites import SUITES


class TestSuiteRegistry:
    def test_all_suites_registered(self) -> None:
        expected = {"basic", "throughput", "agent", "tools", "all"}
        assert set(SUITES.keys()) == expected

    def test_suite_functions_callable(self) -> None:
        for suite_name, tests in SUITES.items():
            for fn in tests:
                assert callable(fn), f"{fn} in {suite_name} is not callable"
                assert asyncio.iscoroutinefunction(fn), (
                    f"{fn.__name__} in {suite_name} is not async"
                )

    def test_all_suite_is_union(self) -> None:
        non_all = [fn for name, fns in SUITES.items() if name != "all" for fn in fns]
        assert set(SUITES["all"]) == set(non_all)

    def test_suite_functions_accept_runner(self) -> None:
        for suite_name, tests in SUITES.items():
            for fn in tests:
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                assert len(params) >= 1, (
                    f"{fn.__name__} in {suite_name} must accept at least one param (runner)"
                )

    def test_no_empty_suites(self) -> None:
        for suite_name, tests in SUITES.items():
            assert len(tests) > 0, f"Suite {suite_name!r} is empty"
