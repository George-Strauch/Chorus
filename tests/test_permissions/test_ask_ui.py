"""Tests for chorus.permissions.ask_ui — Discord permission ask UI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chorus.permissions.ask_ui import PermissionAskView


class TestAskViewAllowSetsValueTrue:
    @pytest.mark.asyncio
    async def test_allow_button_sets_value_true(self) -> None:
        view = PermissionAskView(requester_id=12345)
        interaction = MagicMock()
        interaction.user.id = 12345
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        # Call the allow callback directly (self is already bound)
        await view.allow.callback(interaction)

        assert view.value is True

    @pytest.mark.asyncio
    async def test_allow_sends_ephemeral_response(self) -> None:
        view = PermissionAskView(requester_id=12345)
        interaction = MagicMock()
        interaction.user.id = 12345
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.allow.callback(interaction)

        interaction.response.send_message.assert_called_once_with("Approved.", ephemeral=True)


class TestAskViewDenySetsValueFalse:
    @pytest.mark.asyncio
    async def test_deny_button_sets_value_false(self) -> None:
        view = PermissionAskView(requester_id=12345)
        interaction = MagicMock()
        interaction.user.id = 12345
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.deny.callback(interaction)

        assert view.value is False

    @pytest.mark.asyncio
    async def test_deny_sends_ephemeral_response(self) -> None:
        view = PermissionAskView(requester_id=12345)
        interaction = MagicMock()
        interaction.user.id = 12345
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.deny.callback(interaction)

        interaction.response.send_message.assert_called_once_with("Denied.", ephemeral=True)


class TestAskViewTimeoutLeavesValueNone:
    @pytest.mark.asyncio
    async def test_initial_value_is_none(self) -> None:
        view = PermissionAskView(requester_id=12345)
        assert view.value is None

    @pytest.mark.asyncio
    async def test_timeout_disables_buttons(self) -> None:
        view = PermissionAskView(requester_id=12345)
        await view.on_timeout()
        for child in view.children:
            assert child.disabled is True  # type: ignore[union-attr]


class TestAskViewInteractionCheck:
    @pytest.mark.asyncio
    async def test_rejects_wrong_user(self) -> None:
        view = PermissionAskView(requester_id=12345)
        interaction = MagicMock()
        interaction.user.id = 99999

        result = await view.interaction_check(interaction)
        assert result is False

    @pytest.mark.asyncio
    async def test_allows_requester(self) -> None:
        view = PermissionAskView(requester_id=12345)
        interaction = MagicMock()
        interaction.user.id = 12345

        result = await view.interaction_check(interaction)
        assert result is True


class TestAskViewDisablesButtonsOnTimeout:
    @pytest.mark.asyncio
    async def test_all_children_disabled_after_timeout(self) -> None:
        view = PermissionAskView(requester_id=12345)
        # Verify buttons exist and are initially enabled
        assert len(list(view.children)) > 0
        for child in view.children:
            assert child.disabled is False  # type: ignore[union-attr]

        await view.on_timeout()

        for child in view.children:
            assert child.disabled is True  # type: ignore[union-attr]
        # Value should still be None (not set)
        assert view.value is None
