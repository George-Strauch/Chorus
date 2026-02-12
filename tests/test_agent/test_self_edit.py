"""Tests for chorus.agent.self_edit — agent self-modification."""

import pytest


class TestSelfEditSystemPrompt:
    def test_updates_agent_json(self) -> None:
        pytest.skip("Not implemented yet — TODO 010")

    def test_logs_to_audit(self) -> None:
        pytest.skip("Not implemented yet — TODO 010")


class TestSelfEditDocs:
    def test_writes_doc_file(self) -> None:
        pytest.skip("Not implemented yet — TODO 010")

    def test_path_must_be_within_docs(self) -> None:
        pytest.skip("Not implemented yet — TODO 010")


class TestSelfEditPermissions:
    def test_role_gating(self) -> None:
        pytest.skip("Not implemented yet — TODO 010")

    def test_cannot_escalate_beyond_user_role(self) -> None:
        pytest.skip("Not implemented yet — TODO 010")


class TestSelfEditModel:
    def test_updates_model_in_config(self) -> None:
        pytest.skip("Not implemented yet — TODO 010")

    def test_rejects_unavailable_model(self) -> None:
        pytest.skip("Not implemented yet — TODO 010")
