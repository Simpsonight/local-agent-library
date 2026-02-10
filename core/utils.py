"""Utilities: web scraper, file operations."""

from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def scrape_url(url: str, auth: tuple[str, str] | None = None) -> str:
    """Fetch a URL and return clean text content (tags stripped)."""
    headers = {"User-Agent": "LAL/1.0"}
    if auth:
        # Follow redirects manually to preserve auth across host changes
        # (requests strips the Authorization header on cross-host redirects)
        for _ in range(10):
            resp = requests.get(url, timeout=15, headers=headers, auth=auth, allow_redirects=False)
            if resp.is_redirect and "Location" in resp.headers:
                url = urljoin(resp.url, resp.headers["Location"])
                continue
            break
    else:
        resp = requests.get(url, timeout=15, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def read_file(path: str) -> str:
    """Read a file as UTF-8 text."""
    with open(path, encoding="utf-8") as f:
        return f.read()
