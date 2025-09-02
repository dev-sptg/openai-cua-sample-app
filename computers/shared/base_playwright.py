# base_playwright.py
from __future__ import annotations

import os
import time
import base64
import datetime as _dt
from typing import List, Dict, Optional, Tuple
from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext
from utils import check_blocklisted_url

# -----------------------------------------------------------------------------
# Behavior toggles (env)
# -----------------------------------------------------------------------------
# Opt-in focus: set FOCUS_FRONT=1 if you want Chromium to jump to foreground on tab changes
FOCUS_FRONT = os.getenv("FOCUS_FRONT", "0") not in ("0", "false", "False", "")

# Optional delay after each stitched-frame (ms) so playback is slower
STITCH_SLEEP_MS = int(os.getenv("STITCH_SLEEP_MS", "0"))

# Optional minimum interval between appended frames (ms) – throttles bursty frames
MIN_FRAME_INTERVAL_MS = int(os.getenv("VIDEO_MIN_FRAME_INTERVAL_MS", "0"))


# Optional screencast (stitched from screenshots) — requires imageio/imageio-ffmpeg
try:
    import imageio  # pip install imageio imageio-ffmpeg
except Exception:
    imageio = None

RECORD_SCREENCAST = os.getenv("RECORD_SCREENCAST", "0") not in ("0", "false", "False", "")
VIDEO_DIR = os.getenv("VIDEO_DIR", "./runs")
VIDEO_BASENAME = os.getenv("VIDEO_BASENAME", "session")
VIDEO_FPS = float(os.getenv("VIDEO_FPS", "4"))

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
# 1x1 transparent PNG (base64) used if we can't produce a real screenshot
_BLANK_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)

def _is_closed(obj) -> bool:
    """Best-effort check whether a Playwright object is no longer usable."""
    try:
        if obj is None:
            return True
        if hasattr(obj, "is_closed"):
            return obj.is_closed()
        if hasattr(obj, "is_connected"):
            return not obj.is_connected()
        return False
    except Exception:
        return True

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


def _safe_eval_frame(frame, script, arg=None):
    try:
        if arg is None:
            return frame.evaluate(script)
        return frame.evaluate(script, arg)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Base "Computer" for Playwright
# -----------------------------------------------------------------------------
class BasePlaywrightComputer:
    """
    Features:
      - tracks BrowserContext and Pages
      - handles popups/new tabs and active-tab switching
      - exposes standard computer actions (click/scroll/type/drag/keypress)
      - tab helpers (list_tabs/switch_tab/new_tab/close_tab/next_tab/prev_tab)
      - DOM overlays: cursor dot + reasoning bubble + timestamp watermark (non-interactive)
      - gracefully recovers when tabs/context/browser are closed
      - screenshot never crashes (blank PNG fallback)
      - optional screencast from screenshots (RECORD_SCREENCAST=1)
      - prints paths of both video types on shutdown
    """

    # ---------- public info ----------
    def get_environment(self) -> str:
        return "browser"

    def get_dimensions(self) -> Tuple[int, int]:
        return (1366, 1024)

    # ---------- lifecycle ----------
    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._home_url: Optional[str] = None   # remembered to recreate a page when all tabs are gone
        self._bound: bool = False

        # overlays
        self._overlay_text: Optional[str] = None
        self._last_pointer: Tuple[int, int] = (20, 20)
        self._last_frame_ts = 0.0

        # screencast (stitched)
        self._video_writer = None
        self._video_path = None

        # shutdown guard
        self._shutting_down: bool = False

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._browser, self._page = self._get_browser_and_page()
        self._context = self._page.context

        # overlay init for every navigation in this context
        self._install_overlay_init_script(self._context)

        self._bind_everything(self._page)

        # default home for recovery
        self._home_url = self._home_url or "about:blank"

        # prepare screencast path (lazy writer)
        if RECORD_SCREENCAST and imageio is not None:
            os.makedirs(VIDEO_DIR, exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            self._video_path = os.path.join(VIDEO_DIR, f"{VIDEO_BASENAME}-{ts}.mp4")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # enter shutdown mode (suppresses reopen during teardown)
        self._shutting_down = True

        # snapshot pages to resolve native video paths after close
        _pages_snapshot = []
        try:
            if self._context:
                _pages_snapshot = list(self._context.pages)
        except Exception:
            pass

        # robust teardown: ignore races if driver already closed
        try:
            if getattr(self, "_page", None) and not self._page.is_closed():
                try: self._page.close()
                except Exception: pass
        except Exception:
            pass
        try:
            if getattr(self, "_context", None):
                try: self._context.close()
                except Exception: pass
        except Exception:
            pass
        try:
            if getattr(self, "_browser", None):
                try: self._browser.close()
                except Exception: pass
        except Exception:
            pass

        # stitched MP4 (Approach A) — close writer + print path + index
        try:
            if self._video_writer is not None:
                try: self._video_writer.close()
                except Exception: pass
            if self._video_path:
                print(f"[Video] Screencast (stitched): {self._video_path}")
                try:
                    os.makedirs(VIDEO_DIR, exist_ok=True)
                    with open(os.path.join(VIDEO_DIR, "index.txt"), "a") as f:
                        f.write(self._video_path + "\n")
                except Exception:
                    pass
        except Exception:
            pass

        # native PW videos (Approach B) — list after context close
        try:
            native_paths = []
            for p in _pages_snapshot:
                try:
                    v = getattr(p, "video", None)
                    if v:
                        path = v.path()  # available after close
                        if path:
                            native_paths.append(path)
                except Exception:
                    pass
            if native_paths:
                print("[Video] Playwright native recordings:")
                for vp in native_paths:
                    print(f"  - {vp}")
                pw_dir = os.getenv("PW_VIDEO_DIR", "./runs/videos")
                try:
                    os.makedirs(pw_dir, exist_ok=True)
                    with open(os.path.join(pw_dir, "index.txt"), "a") as f:
                        for vp in native_paths:
                            f.write(vp + "\n")
                except Exception:
                    pass
        except Exception:
            pass

        # stop Playwright last
        try:
            if getattr(self, "_playwright", None):
                self._playwright.stop()
        except Exception:
            pass
        return False

    def ensure_overlay(self) -> None:
        """Public wrapper to guarantee overlays are installed on the current page."""
        try:
            self._ensure_active_page()
            self._ensure_overlay_dom(self._page)
        except Exception:
            pass

    # ---------- overlay installation ----------
    def _install_overlay_init_script(self, context: BrowserContext) -> None:
        """
        Injects overlay helpers that will run before any page scripts.
        We keep content minimal and CSP-friendly; we still do runtime fallback below.
        """
        js = r"""
          (() => {
            if (window.__cuaOverlayInstalled) return;
            window.__cuaOverlayInstalled = true;
            // Create nodes in a tiny, CSP-friendly way (no eval strings later)
            const style = document.createElement('style');
            style.id = 'cua-overlay-style';
            style.textContent = [
              '#cua-pointer,#cua-reason,#cua-time{position:fixed;z-index:2147483647;pointer-events:none}',
              '#cua-pointer{width:16px;height:16px;border-radius:50%;border:2px solid #3B82F6;background:rgba(59,130,246,.25);',
              'transform:translate(-50%,-50%);box-shadow:0 0 8px rgba(59,130,246,.6)}',
              '#cua-reason{right:10px;bottom:42px;max-width:580px;background:rgba(0,0,0,.70);color:#fff;',
              'padding:8px 10px;border-radius:10px;font:12px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;white-space:pre-wrap}',
              '#cua-time{right:10px;bottom:10px;padding:4px 8px;border-radius:8px;background:rgba(17,17,17,.65);color:#fff;',
              'font:12px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}'
            ].join('');
            document.documentElement.appendChild(style);

            const ptr=document.createElement('div'); ptr.id='cua-pointer'; ptr.style.left='20px'; ptr.style.top='20px';
            const bubble=document.createElement('div'); bubble.id='cua-reason'; bubble.textContent='';
            const ts=document.createElement('div'); ts.id='cua-time'; ts.textContent='';
            document.documentElement.appendChild(ptr);
            document.documentElement.appendChild(bubble);
            document.documentElement.appendChild(ts);

            window.cuaOverlay = {
              setReason: (t) => { bubble.textContent = t || ''; },
              setPointer: (x,y) => { ptr.style.left=(x||0)+'px'; ptr.style.top=(y||0)+'px'; },
              setTime: (t) => { ts.textContent = t || ''; },
            };
          })();
        """
        try:
            context.add_init_script(js)
        except Exception:
            pass

    def _ensure_overlay_dom(self, page: Page) -> None:
        """Ensure overlay exists in the main page AND in all child frames (no reload)."""
        if not page or page.is_closed():
            return

        JS_INSTALL = """
            (() => {
              if (window.__cuaOverlayInstalled) return true;

              // Add CSS (idempotent)
              if (!document.getElementById('cua-overlay-style')) {
                const style = document.createElement('style');
                style.id = 'cua-overlay-style';
                style.textContent =
                  '#cua-pointer,#cua-reason,#cua-time{position:fixed;z-index:2147483647;pointer-events:none}' +
                  '#cua-pointer{width:16px;height:16px;border-radius:50%;border:2px solid #3B82F6;' +
                  'background:rgba(59,130,246,.25);transform:translate(-50%,-50%);' +
                  'box-shadow:0 0 8px rgba(59,130,246,.6)}' +
                  '#cua-reason{right:10px;bottom:42px;max-width:520px;background:rgba(0,0,0,.70);color:#fff;' +
                  'padding:8px 10px;border-radius:10px;font:12px/1.45 -apple-system,BlinkMacSystemFont,\\'Segoe UI\\',' +
                  'Roboto,Helvetica,Arial,sans-serif;white-space:pre-wrap}' +
                  '#cua-time{right:10px;bottom:10px;padding:4px 8px;border-radius:8px;background:rgba(17,17,17,.65);' +
                  'color:#fff;font:12px/1.35 -apple-system,BlinkMacSystemFont,\\'Segoe UI\\',' +
                  'Roboto,Helvetica,Arial,sans-serif}';
                document.documentElement.appendChild(style);
              }

              let ptr = document.getElementById('cua-pointer');
              if (!ptr) {
                ptr = document.createElement('div');
                ptr.id = 'cua-pointer';
                ptr.style.left = '20px';
                ptr.style.top  = '20px';
                document.documentElement.appendChild(ptr);
              }

              let bubble = document.getElementById('cua-reason');
              if (!bubble) {
                bubble = document.createElement('div');
                bubble.id = 'cua-reason';
                bubble.textContent = '';
                document.documentElement.appendChild(bubble);
              }

              let ts = document.getElementById('cua-time');
              if (!ts) {
                ts = document.createElement('div');
                ts.id = 'cua-time';
                ts.textContent = '';
                document.documentElement.appendChild(ts);
              }

              window.cuaOverlay = {
                setReason: (t) => { bubble.textContent = t || ''; },
                setPointer: (x, y) => { ptr.style.left = (x||0)+'px'; ptr.style.top = (y||0)+'px'; },
                setTime:   (t) => { ts.textContent = t || ''; },
              };
              window.__cuaOverlayInstalled = true;
              return true;
            })();
        """

        # main frame
        _safe_eval_frame(page.main_frame, JS_INSTALL)

        # child frames (some apps render full UI in a big iframe)
        for fr in page.frames:
            _safe_eval_frame(fr, JS_INSTALL)

        # Last-chance fallback: add CSS via add_style_tag then create nodes via evaluate.
        try:
            page.add_style_tag(content=
                               "#cua-pointer,#cua-reason,#cua-time{position:fixed;z-index:2147483647;pointer-events:none}"
                               "#cua-pointer{width:16px;height:16px;border-radius:50%;border:2px solid #3B82F6;"
                               "background:rgba(59,130,246,.25);transform:translate(-50%,-50%);box-shadow:0 0 8px rgba(59,130,246,.6)}"
                               "#cua-reason{right:10px;bottom:42px;max-width:520px;background:rgba(0,0,0,.70);color:#fff;"
                               "padding:8px 10px;border-radius:10px;font:12px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;white-space:pre-wrap}"
                               "#cua-time{right:10px;bottom:10px;padding:4px 8px;border-radius:8px;background:rgba(17,17,17,.65);color:#fff;"
                               "font:12px/1.35 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}"
                               )
            page.evaluate("""
              (() => {
                if (!document.getElementById('cua-pointer')) {
                  const ptr=document.createElement('div'); ptr.id='cua-pointer'; ptr.style.left='20px'; ptr.style.top='20px';
                  document.documentElement.appendChild(ptr);
                }
                if (!document.getElementById('cua-reason')) {
                  const bubble=document.createElement('div'); bubble.id='cua-reason'; bubble.textContent='';
                  document.documentElement.appendChild(bubble);
                }
                if (!document.getElementById('cua-time')) {
                  const ts=document.createElement('div'); ts.id='cua-time'; ts.textContent='';
                  document.documentElement.appendChild(ts);
                }
                if (!window.cuaOverlay) {
                  const bubble = document.getElementById('cua-reason');
                  const ptr = document.getElementById('cua-pointer');
                  const ts = document.getElementById('cua-time');
                  window.cuaOverlay = {
                    setReason:(t)=>{bubble.textContent=t||''},
                    setPointer:(x,y)=>{ptr.style.left=(x||0)+'px';ptr.style.top=(y||0)+'px'},
                    setTime:(t)=>{ts.textContent=t||''}
                  };
                  window.__cuaOverlayInstalled = true;
                }
              })();
            """)
        except Exception:
            pass

    def _set_overlay_reason(self, text: Optional[str]) -> None:
        self._overlay_text = text or ""
        if self._shutting_down or not self._page or self._page.is_closed():
            return
        try:
            self._ensure_overlay_dom(self._page)
            script = "t => window.cuaOverlay && window.cuaOverlay.setReason(t)"
            self._page.evaluate(script, self._overlay_text)
            for fr in self._page.frames:
                _safe_eval_frame(fr, script, self._overlay_text)
        except Exception:
            pass

    def set_overlay_text(self, text: Optional[str]) -> None:
        """Public method for Agent to set reasoning bubble text."""
        self._set_overlay_reason(text)

    def wait_for_selector(self, selector: str, timeout_ms: int = 4000, **_):
        self._ensure_active_page()
        self._page.wait_for_selector(selector, state="visible", timeout=timeout_ms)

    def click_selector(self, selector: str, timeout_ms: int = 4000, **_):
        self._ensure_active_page()
        self._page.locator(selector).first.click(timeout=timeout_ms)

    def click_text(self, text: str, exact: bool = False, timeout_ms: int = 4000, **_):
        self._ensure_active_page()
        sel = f'text="{text}"' if exact else f"text={text}"
        self._page.locator(sel).first.click(timeout=timeout_ms)

    def type_selector(self, selector: str, text: str, clear: bool = True,
                      delay_ms: int = 0, timeout_ms: int = 4000, **_):
        self._ensure_active_page()
        loc = self._page.locator(selector).first
        if clear:
            loc.fill("", timeout=timeout_ms)
        if delay_ms:
            self._page.keyboard.type(text, delay=delay_ms)
        else:
            loc.type(text, timeout=timeout_ms)

    def reload(self, **_):
        self._ensure_active_page()
        self._page.reload()

    def _update_cursor_dom(self, x: int, y: int) -> None:
        self._last_pointer = (x, y)
        if self._shutting_down or not self._page or self._page.is_closed():
            return
        try:
            self._ensure_overlay_dom(self._page)
            # small clamp so it never hides under the time bubble
            script = """
              (xy) => {
                const [x,y] = xy;
                const yy = Math.max(0, Math.min(window.innerHeight - 60, y)); // keep ~60px off bottom
                const el = window.cuaOverlay ? document.getElementById('cua-pointer') : null;
                if (window.cuaOverlay && el) window.cuaOverlay.setPointer(x, yy);
              }
            """
            self._page.evaluate(script, [x, y])
            for fr in self._page.frames:
                _safe_eval_frame(fr, script, [x, y])
        except Exception:
            pass

    def _update_timestamp_dom(self) -> None:
        """Update the bottom-right watermark to 'Tue, Sep 2 • 05:23:10PM' (no ms)."""
        if self._shutting_down or not self._page or self._page.is_closed():
            return
        try:
            # Short date: Tue, Sep 2
            now = _dt.datetime.now()
            date_str = f"{now.strftime('%a')}, {now.strftime('%b')} {now.day}"
            # Short time: 05:23:10PM  (no space before AM/PM)
            time_str = now.strftime("%I:%M:%S%p")
            stamp = f"{date_str} • {time_str}"

            # Ensure overlay exists and broadcast to all frames
            self._ensure_overlay_dom(self._page)
            script = "t => window.cuaOverlay && window.cuaOverlay.setTime(t)"
            self._page.evaluate(script, stamp)
            for fr in self._page.frames:
                _safe_eval_frame(fr, script, stamp)
        except Exception:
            pass

    # ---------- binding & events ----------
    def _bind_everything(self, page: Page) -> None:
        """Bind routing, popup/page listeners, and close handlers for recovery + overlay."""
        # Intercept network: blocklisted domains raise → abort the route
        def handle_route(route, request):
            url = request.url
            try:
                check_blocklisted_url(url)  # raises ValueError if blocked
            except ValueError:
                print(f"Flagging blocked domain: {url}")
                try: route.abort()
                except Exception: pass
            else:
                try: route.continue_()
                except Exception: pass

        try:
            page.unroute("**/*")  # avoid duplicate handlers if re-binding
        except Exception:
            pass
        page.route("**/*", handle_route)

        # Ensure overlay exists now
        self._ensure_overlay_dom(page)
        try:
            now = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            page.evaluate("t => window.cuaOverlay && window.cuaOverlay.setTime(t)", now)
        except Exception:
            pass

        if self._overlay_text:
            self._set_overlay_reason(self._overlay_text)

        # New tab via window.open/target=_blank
        def _on_popup(popup: Page):
            if FOCUS_FRONT:
                try: popup.bring_to_front()
                except Exception: pass
            self._bind_page(popup)
            self._page = popup
            self._stabilize(popup)
            self._update_timestamp_dom()

        page.on("popup", _on_popup)

        # New page from context (also catches some popup cases)
        def _on_new_page(p: Page):
            if FOCUS_FRONT:
                try: p.bring_to_front()
                except Exception: pass
            self._bind_page(p)
            self._page = p
            self._stabilize(p)
            self._update_timestamp_dom()

        def _on_page_close(p: Page):
            if self._shutting_down:
                return
            if p is self._page:
                self._ensure_active_page()
                self._update_timestamp_dom()

        # Bind context/page listeners
        try:
            page.context.on("page", _on_new_page)
        except Exception:
            pass
        try:
            page.on("close", _on_page_close)
        except Exception:
            pass

        self._bound = True

    def _bind_page(self, page: Page) -> None:
        """Bind popup handler on a given page (used by _bind_everything)."""
        self._ensure_overlay_dom(page)
        def _on_popup(popup: Page):
            if FOCUS_FRONT:
                try: popup.bring_to_front()
                except Exception: pass
            self._bind_page(popup)
            self._page = popup
            self._stabilize(popup)
            self._update_timestamp_dom()
        page.on("popup", _on_popup)

    # ---------- recovery ----------
    def _stabilize(self, p: Page) -> None:
        try:
            p.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass

    def _active_pages(self) -> List[Page]:
        return [p for p in (self._context.pages if self._context else []) if not p.is_closed()]

    def _ensure_active_page(self) -> None:
        """
        Guarantee browser/context/page exist and are open; recreate if needed.
        This is called by every action & screenshot so user tab closes don't crash us.
        """
        if self._shutting_down:
            return

        # Browser gone → full reopen
        if _is_closed(self._browser):
            self._browser, self._page = self._reopen_browser_and_page()
            self._context = self._page.context
            self._install_overlay_init_script(self._context)
            self._bind_everything(self._page)
            if self._home_url and self._home_url != "about:blank":
                try: self._page.goto(self._home_url)
                except Exception: pass
            self._stabilize(self._page)
            return

        # Context gone → create new context/page inside existing browser
        if _is_closed(self._context):
            width, height = self.get_dimensions()
            self._context = self._browser.new_context(viewport={"width": width, "height": height})
            self._install_overlay_init_script(self._context)
            self._page = self._context.new_page()
            self._bind_everything(self._page)
            if self._home_url and self._home_url != "about:blank":
                try: self._page.goto(self._home_url)
                except Exception: pass
            self._stabilize(self._page)
            return

        # Page gone → switch to another or create one
        if _is_closed(self._page):
            pages = [p for p in self._context.pages if not p.is_closed()]
            if pages:
                self._page = pages[-1]
                self._bind_everything(self._page)
                self._stabilize(self._page)
            else:
                self._page = self._context.new_page()
                self._bind_everything(self._page)
                if self._home_url and self._home_url != "about:blank":
                    try: self._page.goto(self._home_url)
                    except Exception: pass
                self._stabilize(self._page)

        # keep overlay state consistent on the active page
        try:
            self._ensure_overlay_dom(self._page)
            if self._overlay_text:
                self._page.evaluate("t => window.cuaOverlay && window.cuaOverlay.setReason(t)", self._overlay_text)
            x, y = self._last_pointer
            self._page.evaluate("(xy)=>window.cuaOverlay && window.cuaOverlay.setPointer(xy[0], xy[1])", [x, y])
        except Exception:
            pass

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
                try: target.bring_to_front()
                except Exception: pass
            self._page = target
            self._stabilize(self._page)
            self._ensure_active_page()

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
            try: p.goto(url)
            except Exception: pass
        if FOCUS_FRONT:
            try: p.bring_to_front()
            except Exception: pass
        self._bind_page(p)
        self._page = p
        self._stabilize(p)
        self._ensure_active_page()

    def close_tab(self, index: Optional[int] = None) -> None:
        pages = self._active_pages()
        target = self._page if index is None else (pages[index] if 0 <= index < len(pages) else None)
        if target and not target.is_closed():
            try: target.close()
            except Exception: pass
            self._ensure_active_page()

    # ---------- screencast ----------
    def _append_to_video(self, png_bytes: bytes) -> None:
        if not (RECORD_SCREENCAST and imageio and self._video_path):
            return
        try:
            # throttle if requested
            if MIN_FRAME_INTERVAL_MS > 0:
                now = time.monotonic() * 1000.0
                if self._last_frame_ts and (now - self._last_frame_ts) < MIN_FRAME_INTERVAL_MS:
                    return
                self._last_frame_ts = now

            WARMUP_DUP = int(os.getenv("VIDEO_WARMUP_DUP_FRAMES", "0"))
            if self._video_writer is None:
                os.makedirs(os.path.dirname(self._video_path), exist_ok=True)
                # prevent resizing; accept non-macroblock sizes
                self._video_writer = imageio.get_writer(self._video_path, fps=VIDEO_FPS, macro_block_size=1)
                if WARMUP_DUP > 0:
                    # append the first frame again immediately
                    frame0 = imageio.v2.imread(png_bytes, format="png")
                    for _ in range(WARMUP_DUP):
                        self._video_writer.append_data(frame0)

            frame = imageio.v2.imread(png_bytes, format="png")
            self._video_writer.append_data(frame)
        except Exception:
            pass

    # ---------- core actions ----------
    def screenshot(self) -> str:
        """
        Capture a viewport screenshot. If the active page is gone (user closed tab),
        automatically switch to another tab or recreate a new page. As a last resort,
        return a 1x1 transparent PNG so the protocol stays valid.
        Also appends the frame to an MP4 screencast when RECORD_SCREENCAST=1.
        """
        try:
            self._ensure_active_page()
            self._update_timestamp_dom()
            png_bytes = self._page.screenshot(full_page=False)
            self._append_to_video(png_bytes)  # best-effort
            if STITCH_SLEEP_MS > 0:
                time.sleep(STITCH_SLEEP_MS / 1000.0)
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception:
            # append a blank frame so timeline length stays consistent
            try:
                self._append_to_video(base64.b64decode(_BLANK_PNG_B64))
            except Exception:
                pass
            return _BLANK_PNG_B64

    def click(self, x: int, y: int, button: str = "left") -> None:
        """
        Supports left/right/middle clicks. 'middle' will open link in a new tab in many UIs.
        Also supports 'back'/'forward' pseudo-buttons for nav.
        """
        self._ensure_active_page()
        self._update_cursor_dom(x, y)
        self._update_timestamp_dom()
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
        self._update_cursor_dom(x, y)
        self._update_timestamp_dom()
        self._page.mouse.dblclick(x, y)

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self._ensure_active_page()
        self._update_cursor_dom(x, y)
        self._update_timestamp_dom()
        self._page.mouse.move(x, y)
        self._page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

    def type(self, text: str) -> None:
        self._ensure_active_page()
        self._update_timestamp_dom()
        self._page.keyboard.type(text)

    def move(self, x: int, y: int) -> None:
        self._ensure_active_page()
        self._update_cursor_dom(x, y)
        self._update_timestamp_dom()
        self._page.mouse.move(x, y)

    def wait(self, ms: int = 1000) -> None:
        time.sleep(ms / 1000)
        # keep ticking timestamp during waits so native video shows time advancing
        self._update_timestamp_dom()

    def keypress(self, keys: List[str]) -> None:
        """
        Press a sequence or combination.
        - If looks like chord (modifiers + one non-mod), press 'Meta+Enter', etc.
        - Otherwise press keys one by one.
        """
        self._ensure_active_page()
        self._update_timestamp_dom()
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

        # special-case: many LLMs send Control+R; on macOS Chrome that's Meta+R
        if set(mods) == {"Control"} and nonmods == ["R"]:
            try:
                self._page.reload()
                return
            except Exception:
                # fallback to Meta+R chord
                self._page.keyboard.press("Meta+R")
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
            self._update_cursor_dom(point["x"], point["y"])
            self._update_timestamp_dom()
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
            # keep overlay alive across navigations
            self._ensure_overlay_dom(self._page)
            if self._overlay_text:
                self._page.evaluate("t => window.cuaOverlay && window.cuaOverlay.setReason(t)", self._overlay_text)
            x, y = self._last_pointer
            self._page.evaluate("(xy)=>window.cuaOverlay && window.cuaOverlay.setPointer(xy[0], xy[1])", [x, y])
            self._update_timestamp_dom()
        except Exception as e:
            print(f"Error navigating to {url}: {e}")

    def back(self) -> None:
        try:
            self._ensure_active_page()
            self._page.go_back()
            self._ensure_overlay_dom(self._page)
            self._update_timestamp_dom()
        except Exception:
            pass

    def forward(self) -> None:
        try:
            self._ensure_active_page()
            self._page.go_forward()
            self._ensure_overlay_dom(self._page)
            self._update_timestamp_dom()
        except Exception:
            pass

    # ---------- subclass hooks ----------
    def _get_browser_and_page(self) -> Tuple[Browser, Page]:
        """Subclasses must implement, returning (Browser, Page)."""
        raise NotImplementedError

    def _reopen_browser_and_page(self) -> Tuple[Browser, Page]:
        """Subclasses must implement, returning (Browser, Page) after relaunch."""
        raise NotImplementedError
