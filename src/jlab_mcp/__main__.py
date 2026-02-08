import logging
import sys

from jlab_mcp.server import mcp, start_jupyter_background


def main():
    # Configure logging to stderr so Claude Code can show server status.
    # stdout is reserved for MCP protocol (stdio transport).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    # Submit SLURM job immediately so it's ready by the time the user
    # calls start_new_session.  The background thread runs while the
    # MCP server handles the stdio handshake with Claude Code.
    start_jupyter_background()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
