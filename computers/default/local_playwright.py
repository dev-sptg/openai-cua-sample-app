# computers/local/local_playwright.py
from __future__ import annotations

import os
from typing import Tuple
from playwright.sync_api import Browser, Page
from ..shared.base_playwright import BasePlaywrightComputer


class LocalPlaywrightBrowser(BasePlaywrightComputer):
    """
    Launches a local Chromium instance using Playwright.

    Env toggles:
      - HEADLESS=1            → launch headless (no window)
      - PW_RECORD=1           → enable Playwright native video recording
      - PW_VIDEO_DIR=./runs/videos
      - PW_VIDEO_WIDTH/HEIGHT → override video size
    """

    def __init__(self, headless: bool | None = None):
        super().__init__()
        if headless is None:
            env_val = os.getenv("HEADLESS", "0")
            headless = env_val not in ("0", "false", "False", "")
        self.headless = headless

        # cache config so we can reopen later
        self._record_enabled = os.getenv("PW_RECORD", "0") not in ("0", "false", "False", "")
        self._video_dir = os.getenv("PW_VIDEO_DIR", "./runs/videos")

    def _make_context(self, browser: Browser) -> Tuple[Browser, Page]:
        width, height = self.get_dimensions()
        if self._record_enabled:
            os.makedirs(self._video_dir, exist_ok=True)
            video_w = int(os.getenv("PW_VIDEO_WIDTH", str(width)))
            video_h = int(os.getenv("PW_VIDEO_HEIGHT", str(height)))
            context = browser.new_context(
                viewport={"width": width, "height": height},
                record_video_dir=self._video_dir,
                record_video_size={"width": video_w, "height": video_h},
            )
        else:
            context = browser.new_context(
                viewport={"width": width, "height": height},
            )
        page = context.new_page()
        return browser, page

    def _launch_browser(self) -> Browser:
        width, height = self.get_dimensions()
        launch_args = [
            f"--window-size={width},{height}",
            "--disable-extensions",
            "--disable-features=Translate,PrivacySandboxSettings4",
            "--disable-dev-shm-usage",
            "--no-default-browser-check",
            "--no-first-run",
        ]
        return self._playwright.chromium.launch(
            headless=self.headless,
            args=launch_args,
        )

    def _get_browser_and_page(self) -> Tuple[Browser, Page]:
        browser = self._launch_browser()
        return self._make_context(browser)

    # >>> NEW: allow the base class to fully recover after user closes everything
    def _reopen_browser_and_page(self) -> Tuple[Browser, Page]:
        # playwright is already started; just relaunch the browser and context
        browser = self._launch_browser()
        return self._make_context(browser)
