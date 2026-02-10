"""Clipboard copy — platform-aware (macOS / Linux / Windows)."""

import subprocess
import sys


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    try:
        if sys.platform == "darwin":
            cmd = ["pbcopy"]
        elif sys.platform.startswith("linux"):
            cmd = ["xclip", "-selection", "clipboard"]
        else:
            cmd = ["clip"]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
