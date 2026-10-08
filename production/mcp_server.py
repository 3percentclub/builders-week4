"""Expose the shelf as an MCP server so any agent (Claude Desktop, Cursor, Claude Code) can search it.

    python production/mcp_server.py      # stdio transport

Week 3 gave your agent hands (tools). This gives it a library card.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from shelf import get_shelf

mcp = FastMCP("midnight-rental")


@mcp.tool()
def search_shelf(query: str, k: int = 3) -> list[dict]:
    """Search the Midnight Rental horror shelf. Works for vague plot descriptions
    and exact names (actors, places, tape numbers). Returns up to k tapes, best first."""
    return [h.as_dict() for h in get_shelf().search(query, k=max(1, min(k, 10)))]


@mcp.tool()
def get_tape(tape: str) -> dict:
    """Get the full record for one tape by its 4-digit number, e.g. "1031"."""
    movie = get_shelf().get_tape(tape)
    return movie if movie else {"error": f"No tape #{tape} on the shelf."}


if __name__ == "__main__":
    mcp.run()
