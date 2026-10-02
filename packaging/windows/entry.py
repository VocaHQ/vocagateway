"""PyInstaller entry point for the frozen ``vocagateway.exe`` build.

The packaged executable runs the server directly — the same thing
``uv run vocagateway`` does from a source checkout. Token, status, and
diagnostics are available in the WebUI, which the installer links into the
Start Menu.
"""

from __future__ import annotations

from app.cli import serve

if __name__ == "__main__":
    serve()
