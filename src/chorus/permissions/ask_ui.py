"""Discord UI for permission ask prompts.

When the permission engine returns :data:`ASK`, this module provides the
Discord view (Allow / Deny buttons) that suspends the tool loop coroutine
via ``view.wait()`` until the user responds or the view times out.
"""

from __future__ import annotations

import discord


class PermissionAskView(discord.ui.View):
    """A two-button view for Allow / Deny permission prompts.

    Parameters
    ----------
    requester_id:
        The Discord user ID allowed to interact with the buttons.
    timeout:
        Seconds before the view auto-expires.  Timeout = deny.
    """

    def __init__(self, requester_id: int, timeout: float = 120) -> None:
        super().__init__(timeout=timeout)
        self.requester_id = requester_id
        self.value: bool | None = None  # None = timed out

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Only the user who triggered the agent can respond."""
        return interaction.user.id == self.requester_id

    @discord.ui.button(label="Allow", style=discord.ButtonStyle.green)
    async def allow(
        self, interaction: discord.Interaction, button: discord.ui.Button[PermissionAskView]
    ) -> None:
        await interaction.response.send_message("Approved.", ephemeral=True)
        self.value = True
        self.stop()

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
    async def deny(
        self, interaction: discord.Interaction, button: discord.ui.Button[PermissionAskView]
    ) -> None:
        await interaction.response.send_message("Denied.", ephemeral=True)
        self.value = False
        self.stop()

    async def on_timeout(self) -> None:
        """Disable buttons when the view times out."""
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]


async def ask_user_permission(
    channel: discord.TextChannel,
    requester_id: int,
    action_string: str,
    tool_name: str,
    arguments: str,
) -> bool:
    """Send a permission prompt to *channel* and wait for a response.

    Returns ``True`` if the user clicked Allow, ``False`` otherwise
    (including timeout).
    """
    embed = discord.Embed(
        title="Permission Required",
        description=f"`{action_string}`",
        color=discord.Color.yellow(),
    )
    embed.add_field(name="Tool", value=tool_name, inline=True)
    embed.add_field(name="Arguments", value=f"```\n{arguments}\n```", inline=False)

    view = PermissionAskView(requester_id=requester_id, timeout=120)
    msg = await channel.send(embed=embed, view=view)

    timed_out = await view.wait()

    # Disable buttons on the message after resolution
    for child in view.children:
        child.disabled = True  # type: ignore[attr-defined]
    await msg.edit(view=view)

    if timed_out or view.value is None:
        return False
    return view.value
