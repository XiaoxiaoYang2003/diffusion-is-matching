#!/usr/bin/env python3
"""Screenshot the local blog in viewport-height slices for visual review.

    python3 tools/shoot.py [--width 1440] [--out /tmp/shots]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8877/index.html")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--out", default="/tmp/shots")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_timeout(800)
        total = page.evaluate("document.documentElement.scrollHeight")
        n = 0
        y = 0
        while y < total:
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(150)
            page.screenshot(path=str(out / f"{n:02d}.png"))
            n += 1
            y += args.height - 80
        browser.close()
    print(f"{n} slices, page height {total}px -> {out}")


if __name__ == "__main__":
    main()
