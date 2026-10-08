"""The shelf as an MCP server, so any MCP client (Claude Desktop, Cursor, your Week 3 agent) can search it.

    python part2/mcp_server.py      # stdio transport

No API key needed: search runs on local embeddings by default.
Endings stay hidden unless the server operator sets SHELF_ALLOW_SPOILERS=1. A connecting client
cannot turn spoilers on by itself, because the caller is not the one who should decide that.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from shelf import get_shelf

ALLOW_SPOILERS = os.environ.get("SHELF_ALLOW_SPOILERS") == "1"
MAX_K = 10
MAX_QUERY_CHARS = 300

mcp = FastMCP("midnight-rental")


@mcp.tool()
def search_shelf(query: str, k: int = 3) -> list[dict]:
    """Search the Midnight Rental horror shelf by description, title, director, or cast. Returns up to k tapes."""
    query = query.strip()[:MAX_QUERY_CHARS]
    if not query:
        return []
    k = max(1, min(int(k), MAX_K))
    return [h.as_dict() for h in get_shelf().search(query, k=k, include_spoilers=ALLOW_SPOILERS)]


@mcp.tool()
def get_tape(tape: str) -> dict:
    """Fetch one tape by its 4-digit number."""
    record = get_shelf().get_tape(tape, include_spoilers=ALLOW_SPOILERS)
    return record or {"error": f"no tape #{tape.strip()} on the shelf"}


if __name__ == "__main__":
    mcp.run()
