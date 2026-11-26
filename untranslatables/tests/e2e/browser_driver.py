"""Headless Chrome / Playwright driver utilities for Translation Explorer E2E UI tests.

Bridges Playwright from ``sys.base_prefix`` site-packages when running inside a
``uv``/venv Python (such as ``/tmp/cpu-test-venv``) sharing the same CPython ABI,
prefers ``/usr/bin/google-chrome`` for fast (~350 ms) deterministic startup,
and aborts external HTTP/HTTPS CDN requests (such as Google Fonts) without
interfering with ``127.0.0.1``/``localhost`` API/SSE traffic or ``file://`` URLs.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
from typing import Any, Iterator

# Matches any external http(s) URL while excluding loopback (127.0.0.1 / localhost)
# so local HTTP APIs, SSE EventSource streams, and file:// assets are untouched.
EXTERNAL_HTTP_RE = re.compile(r"^https?://(?!(127\.0\.0\.1|localhost)(:|/|$))")

CHROME_ARGS = [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
]


def ensure_playwright_importable():
    """Import and return ``playwright.sync_api``.

    If ``playwright`` is not installed in the active virtualenv, appends
    ``sys.base_prefix/lib/pythonX.Y/site-packages`` to the *end* of ``sys.path``
    so virtualenv packages keep priority.
    """
    try:
        from playwright import sync_api

        return sync_api
    except ModuleNotFoundError:
        base_sp = os.path.join(
            sys.base_prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )
        if os.path.isdir(base_sp) and base_sp not in sys.path:
            sys.path.append(base_sp)
        from playwright import sync_api

        return sync_api


def launch_headless_browser(pw: Any):
    """Launch headless Chromium, preferring system ``/usr/bin/google-chrome``."""
    chrome_path = "/usr/bin/google-chrome"
    if not os.path.isfile(chrome_path):
        chrome_path = shutil.which("google-chrome") or ""

    launch_kwargs: dict[str, Any] = {
        "headless": True,
        "args": list(CHROME_ARGS),
    }
    if chrome_path:
        launch_kwargs["executable_path"] = chrome_path

    try:
        return pw.chromium.launch(**launch_kwargs)
    except Exception:
        # Fallback to Playwright's bundled Chromium if system Chrome fails.
        launch_kwargs.pop("executable_path", None)
        return pw.chromium.launch(**launch_kwargs)


@contextlib.contextmanager
def new_isolated_page(
    browser: Any,
    viewport: tuple[int, int] = (1280, 900),
) -> Iterator[tuple[Any, Any, list[str], list[str], list[str]]]:
    """Create an isolated ``BrowserContext`` + ``Page`` with external-URL blocking.

    Yields ``(context, page, page_errors, console_errors, external_requests)``
    and guarantees context teardown in ``finally``.
    """
    context = browser.new_context(
        viewport={"width": int(viewport[0]), "height": int(viewport[1])},
    )
    page_errors: list[str] = []
    console_errors: list[str] = []
    external_requests: list[str] = []

    def _abort_external(route: Any) -> None:
        external_requests.append(route.request.url)
        route.abort()

    context.route(EXTERNAL_HTTP_RE, _abort_external)
    page = context.new_page()

    def _on_pageerror(err: Any) -> None:
        page_errors.append(str(err))

    def _on_console(msg: Any) -> None:
        if msg.type == "error":
            text = msg.text
            loc_url = ""
            with contextlib.suppress(Exception):
                loc = msg.location
                if isinstance(loc, dict):
                    loc_url = str(loc.get("url", ""))
            # Ignore aborted external CDN resource messages triggered by our route blocker.
            if "ERR_FAILED" in text and (
                (loc_url and EXTERNAL_HTTP_RE.match(loc_url))
                or (text.strip() == "Failed to load resource: net::ERR_FAILED" and external_requests)
            ):
                return
            console_errors.append(text)

    page.on("pageerror", _on_pageerror)
    page.on("console", _on_console)

    try:
        yield context, page, page_errors, console_errors, external_requests
    finally:
        with contextlib.suppress(Exception):
            page.close()
        with contextlib.suppress(Exception):
            context.close()


def drag_svg_point(
    page: Any,
    selector: str,
    norm_x: float,
    norm_y: float,
    extent: float = 1.45,
    steps: int = 5,
) -> tuple[float, float]:
    """Deterministically drag the pointer on a compass SVG to ``(norm_x, norm_y)``.

    Maps normalized compass coordinates in ``[-extent, +extent]`` (with ``+y``
    pointing *up* as in mathematical compass coordinates, inverted for SVG screen Y)
    onto the element's bounding box and performs a pointer-captured drag:
    ``mouse.move`` -> ``mouse.down`` -> ``mouse.move(..., steps=steps)`` -> ``mouse.up``.
    """
    loc = page.locator(selector).first
    loc.wait_for(state="visible", timeout=10000)
    box = loc.bounding_box()
    if box is None:
        raise AssertionError(f"Element {selector!r} has no bounding box")

    cx = box["x"] + box["width"] / 2.0
    cy = box["y"] + box["height"] / 2.0

    # Clamp normalized coordinates to stay strictly inside the SVG bounds.
    clamped_x = max(-extent * 0.95, min(extent * 0.95, float(norm_x)))
    clamped_y = max(-extent * 0.95, min(extent * 0.95, float(norm_y)))

    tx = box["x"] + ((clamped_x + extent) / (2.0 * extent)) * box["width"]
    ty = box["y"] + ((extent - clamped_y) / (2.0 * extent)) * box["height"]

    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(tx, ty, steps=max(1, int(steps)))
    page.mouse.up()
    return tx, ty
