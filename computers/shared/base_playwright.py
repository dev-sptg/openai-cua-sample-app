# base_playwright.py
import os
import time
import base64
from typing import List, Dict, Optional, Tuple
from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext
from utils import check_blocklisted_url

# Opt-in focus: set FOCUS_FRONT=1 if you want Chromium to jump to foreground on tab changes
FOCUS_FRONT = os.getenv("FOCUS_FRONT", "0") not in ("0", "false", "False", "")

# 1x1 transparent PNG (base64) used if we can't produce a real screenshot
_BLANK_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)

# mac-friendly normalizer from CUA names → Playwright names
CUA_KEY_TO_PLAYWRIGHT_KEY = {
    # modifiers
    "cmd": "Meta", "command": "Meta", "super": "Meta", "win": "Meta",
    "ctrl": "Control", "control": "Control",
    "alt": "Alt", "option": "Alt", "opt": "Alt",
    "shift": "Shift",

    # navigation / special
    "enter": "Enter", "return": "Enter",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "esc": "Escape", "escape": "Escape",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    "home": "Home", "end": "End",
    "pageup": "PageUp", "pagedown": "PageDown",

    # spacebar
    "space": "Space",

    # optional: numpad
    "numpaddivide": "NumpadDivide",
    "numpadmultiply": "NumpadMultiply",
    "numpadsubtract": "NumpadSubtract",
    "numpadadd": "NumpadAdd",
    "numpaddecimal": "NumpadDecimal",
}
MOD_KEYS = {"Shift", "Control", "Alt", "Meta"}

def _normalize_key(k: str) -> str:
    k = (k or "").strip()
    low = k.lower()
    if low in CUA_KEY_TO_PLAYWRIGHT_KEY:
        return CUA_KEY_TO_PLAYWRIGHT_KEY[low]
    if k == " ":
        return "Space"
    return k  # printable or already Playwright-compatible


class BasePlaywrightComputer:
    """
    Base for Playwright-based "computer" backends.
    Subclasses implement `_get_browser_and_page()` to return (Browser, Page).
    This base:
      - tracks BrowserContext and Pages
      - handles popups/new tabs and active-tab switching
      - exposes standard computer actions (click/scroll/type/…)
      - adds tab helpers (list_tabs/switch_tab/new_tab/close_tab/next_tab/prev_tab)
      - gracefully recovers when tabs are closed; screenshot never crashes
    """

    def get_environment(self) -> str:
        return "browser"

    def get_dimensions(self) -> Tuple[int, int]:
        return (1366, 1024)

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._home_url: Optional[str] = None  # remembered to recreate a page when all tabs are gone

    # ---------- lifecycle ----------

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._browser, self._page = self._get_browser_and_page()
        self._context = self._page.context  # track context for tabs/pages

        # Intercept network: blocklisted domains raise → abort the route
        def handle_route(route, request):
            url = request.url
            try:
                check_blocklisted_url(url)  # raises ValueError if blocked
            except ValueError:
                print(f"Flagging blocked domain: {url}")
                try:
                    route.abort()
                except Exception:
                    pass
            else:
                try:
                    route.continue_()
                except Exception:
                    pass

        self._page.route("**/*", handle_route)

        # Track new pages/tabs and popups
        def _on_new_page(p: Page):
            if FOCUS_FRONT:
                try:
                    p.bring_to_front()
                except Exception:
                    pass
            self._bind_page(p)
            self._page = p  # make newest active
            self._stabilize(p)

        def _on_page_close(p: Page):
            if p is self._page:
                self._ensure_active_page()

        self._context.on("page", _on_new_page)
        self._bind_page(self._page)
        self._page.on("close", _on_page_close)

        # default "home" to recover if everything is closed
        self._home_url = self._home_url or "about:blank"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # robust teardown: ignore races if driver already closed
        try:
            if getattr(self, "_page", None) and not self._page.is_closed():
                try:
                    self._page.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if getattr(self, "_context", None):
                try:
                    self._context.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if getattr(self, "_browser", None):
                try:
                    self._browser.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if getattr(self, "_playwright", None):
                self._playwright.stop()
        except Exception:
            pass
        return False

    def _bind_page(self, page: Page) -> None:
        # when target=_blank/window.open, Playwright emits 'popup'
        def _on_popup(popup: Page):
            if FOCUS_FRONT:
                try:
                    popup.bring_to_front()
                except Exception:
                    pass
            self._bind_page(popup)
            self._page = popup  # switch to the popup by default
            self._stabilize(popup)

        page.on("popup", _on_popup)

    def _stabilize(self, p: Page) -> None:
        # small stabilization after tab switches/popups
        try:
            p.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass

    def _active_pages(self) -> List[Page]:
        return [p for p in (self._context.pages if self._context else []) if not p.is_closed()]

    def _ensure_active_page(self) -> None:
        """Guarantee self._page points to an open Page; recreate one if needed."""
        if self._page and not self._page.is_closed():
            return
        pages = self._active_pages()
        if pages:
            self._page = pages[-1]
            self._stabilize(self._page)
            return
        if self._context:
            p = self._context.new_page()
            try:
                if self._home_url:
                    p.goto(self._home_url)
            except Exception:
                pass
            self._bind_page(p)
            self._page = p
            self._stabilize(self._page)

    # ---------- info helpers ----------

    def get_current_url(self) -> str:
        return self._page.url if self._page else ""

    # ---------- tab helpers ----------

    def list_tabs(self) -> List[Dict[str, str]]:
        res = []
        for i, p in enumerate(self._active_pages()):
            try:
                res.append({"index": str(i), "url": p.url or "", "title": p.title() or ""})
            except Exception:
                res.append({"index": str(i), "url": "", "title": ""})
        return res

    def switch_tab(self, index: Optional[int] = None,
                   url_substr: Optional[str] = None,
                   title_substr: Optional[str] = None) -> None:
        pages = self._active_pages()
        target = None
        if index is not None and 0 <= index < len(pages):
            target = pages[index]
        elif url_substr:
            target = next((p for p in pages if url_substr in (p.url or "")), None)
        elif title_substr:
            target = next((p for p in pages if title_substr.lower() in (p.title() or "").lower()), None)
        if target:
            if FOCUS_FRONT:
                try:
                    target.bring_to_front()
                except Exception:
                    pass
            self._page = target
            self._stabilize(self._page)

    def next_tab(self) -> None:
        pages = self._active_pages()
        if not pages:
            return
        i = pages.index(self._page) if self._page in pages else -1
        self.switch_tab(index=(i + 1) % len(pages))

    def prev_tab(self) -> None:
        pages = self._active_pages()
        if not pages:
            return
        i = pages.index(self._page) if self._page in pages else 1
        self.switch_tab(index=(i - 1) % len(pages))

    def new_tab(self, url: Optional[str] = None) -> None:
        if not self._context:
            return
        p = self._context.new_page()
        if url:
            try:
                p.goto(url)
            except Exception:
                pass
        if FOCUS_FRONT:
            try:
                p.bring_to_front()
            except Exception:
                pass
        self._bind_page(p)
        self._page = p
        self._stabilize(p)

    def close_tab(self, index: Optional[int] = None) -> None:
        pages = self._active_pages()
        target = self._page if index is None else (pages[index] if 0 <= index < len(pages) else None)
        if target and not target.is_closed():
            try:
                target.close()
            except Exception:
                pass
            self._ensure_active_page()

    # ---------- core actions ----------

    def screenshot(self) -> str:
        """
        Capture a viewport screenshot. If the active page is gone (user closed tab),
        automatically switch to another tab or recreate a new page. As a last resort,
        return a 1x1 transparent PNG so the protocol stays valid.
        """
        try:
            self._ensure_active_page()
            png_bytes = self._page.screenshot(full_page=False)
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception:
            return _BLANK_PNG_B64

    def click(self, x: int, y: int, button: str = "left") -> None:
        """
        Supports left/right/middle clicks. 'middle' will open link in a new tab in many UIs.
        Also supports 'back'/'forward' pseudo-buttons for nav.
        """
        self._ensure_active_page()
        if button == "back":
            return self.back()
        if button == "forward":
            return self.forward()
        if button == "wheel":
            return self._page.mouse.wheel(x, y)
        btn = button if button in {"left", "right", "middle"} else "left"
        self._page.mouse.click(x, y, button=btn)

    def double_click(self, x: int, y: int) -> None:
        self._ensure_active_page()
        self._page.mouse.dblclick(x, y)

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self._ensure_active_page()
        self._page.mouse.move(x, y)
        self._page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

    def type(self, text: str) -> None:
        self._ensure_active_page()
        self._page.keyboard.type(text)

    def move(self, x: int, y: int) -> None:
        self._ensure_active_page()
        self._page.mouse.move(x, y)

    def wait(self, ms: int = 1000) -> None:
        time.sleep(ms / 1000)

    def keypress(self, keys: List[str]) -> None:
        """
        Press a sequence or combination.
        - If looks like chord (modifiers + one non-mod), press 'Meta+Enter', etc.
        - Otherwise press keys one by one.
        """
        self._ensure_active_page()
        if not keys:
            return
        mapped = [_normalize_key(k) for k in keys if k]
        mods = [k for k in mapped if k in MOD_KEYS]
        nonmods = [k for k in mapped if k not in MOD_KEYS]

        # chord: Meta/Control/Alt/Shift + ONE key
        if mods and len(nonmods) == 1:
            combo = "+".join(mods + [nonmods[0]])
            self._page.keyboard.press(combo)
            return

        # single key
        if len(mapped) == 1:
            self._page.keyboard.press(mapped[0])
            return

        # sequence
        for k in mapped:
            self._page.keyboard.press(k)

    def drag(self, path: List[Dict[str, int]]) -> None:
        self._ensure_active_page()
        if not path:
            return
        self._page.mouse.move(path[0]["x"], path[0]["y"])
        self._page.mouse.down()
        for point in path[1:]:
            self._page.mouse.move(point["x"], point["y"])
        self._page.mouse.up()

    # ---------- navigation ----------

    def goto(self, url: str) -> None:
        try:
            # remember first non-blank as home for recovery
            if url and url != "about:blank":
                self._home_url = url
            self._ensure_active_page()
            self._page.goto(url)
        except Exception as e:
            print(f"Error navigating to {url}: {e}")

    def back(self) -> None:
        try:
            self._ensure_active_page()
            self._page.go_back()
        except Exception:
            pass

    def forward(self) -> None:
        try:
            self._ensure_active_page()
            self._page.go_forward()
        except Exception:
            pass

    # ---------- subclass hook ----------

    def _get_browser_and_page(self) -> Tuple[Browser, Page]:
        """Subclasses must implement, returning (Browser, Page)."""
        raise NotImplementedError
