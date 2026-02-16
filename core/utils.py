"""Utilities: web scraper, file operations."""

from __future__ import annotations

import ipaddress
import logging
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files that must never be read via /file, regardless of directory.
_BLOCKED_FILENAMES = {".env", ".env.local", ".env.production", ".env.staging"}


# ── SSRF Protection ─────────────────────────────────────


def _is_safe_url(url: str) -> bool:
    """Return True if *url* is safe to fetch (public HTTP(S) only)."""
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Block obvious localhost aliases
        if hostname.lower() in ("localhost", "0.0.0.0"):
            return False

        # Resolve to IP and check for private/internal ranges
        try:
            ip_str = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip_str)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
                return False
        except (socket.gaierror, ValueError):
            return False

        return True
    except Exception:
        return False


def scrape_url(url: str, auth: tuple[str, str] | None = None) -> str:
    """Fetch a URL and return clean text content (tags stripped).

    Raises ``ValueError`` for unsafe URLs (private/internal networks,
    non-HTTP schemes, cloud metadata endpoints).
    """
    if not _is_safe_url(url):
        raise ValueError(
            f"URL blocked: only public HTTP(S) URLs are allowed. "
            f"Cannot access private/internal networks or non-HTTP schemes: {url}"
        )

    headers = {"User-Agent": "LAL/1.0"}
    if auth:
        for _ in range(10):
            resp = requests.get(url, timeout=15, headers=headers, auth=auth, allow_redirects=False)
            if resp.is_redirect and "Location" in resp.headers:
                redirect_url = urljoin(resp.url, resp.headers["Location"])
                if not _is_safe_url(redirect_url):
                    raise ValueError(f"Redirect to unsafe URL blocked: {redirect_url}")
                url = redirect_url
                continue
            break
    else:
        resp = requests.get(url, timeout=15, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


# ── Path Traversal Protection ────────────────────────────


def read_file(path: str) -> str:
    """Read a file as UTF-8 text.

    Raises ``ValueError`` when the resolved path escapes the project tree
    or targets a blocked filename (e.g. ``.env``).
    """
    file_path = Path(path).resolve()

    # Block sensitive filenames regardless of location
    if file_path.name in _BLOCKED_FILENAMES:
        raise ValueError(f"Access denied: reading '{file_path.name}' files is not allowed.")

    # The resolved path must stay within the project root
    try:
        file_path.relative_to(_PROJECT_ROOT)
    except ValueError:
        raise ValueError(
            f"Access denied: path must be within the project directory ({_PROJECT_ROOT}). "
            f"Resolved path: {file_path}"
        )

    with open(file_path, encoding="utf-8") as f:
        return f.read()
