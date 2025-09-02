from collections import deque
import json
from typing import Callable
import re
import inspect
import os
from utils import compress_b64_png_to_jpeg_bytes, upload_image_get_file_id

from computers import Computer
from utils import (
    create_response,
    show_image,
    pp,
    sanitize_message,
    check_blocklisted_url,
)


import re  # add with your imports

def _pair_reasoning_calls(all_items: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Scan the full history once and capture exact adjacency pairs:
      reasoning rs_*  immediately followed by  computer_call cu_*.
    Returns:
      - callid_to_reasoning: {call_id -> reasoning_item}
      - reasoningid_to_call: {reasoning_id -> computer_call_item}
    """
    callid_to_reasoning: dict[str, dict] = {}
    reasoningid_to_call: dict[str, dict] = {}
    for i, it in enumerate(all_items[:-1]):
        if it.get("type") == "reasoning":
            nxt = all_items[i + 1]
            if nxt.get("type") == "computer_call":
                cid = nxt.get("call_id")
                rid = it.get("id")
                if cid and rid:
                    callid_to_reasoning[cid] = it
                    reasoningid_to_call[rid] = nxt
    return callid_to_reasoning, reasoningid_to_call


def _compact_base(all_items: list[dict], keep_images: int = 1, keep_tail: int = 30) -> list[dict]:
    """
    Small base context:
      - latest user 'message'
      - last keep_tail items
      - only the most recent keep_images input_image(s)
    """
    if not all_items:
        return []
    tail = list(all_items[-keep_tail:])

    # ensure latest user message is present
    last_user = next(
        (it for it in reversed(all_items)
         if it.get("type") == "message" and it.get("role") == "user"),
        None
    )
    if last_user and last_user not in tail:
        tail = [last_user] + tail

    # keep only most recent K screenshots
    imgs = [i for i in tail if i.get("type") == "input_image"]
    if len(imgs) > keep_images:
        drop_ids = set(id(i) for i in imgs[:-keep_images])
        tail = [i for i in tail if id(i) not in drop_ids]

    return tail


def _enforce_cua_via_pairs(all_items: list[dict], request: list[dict]) -> list[dict]:
    """
    Strict request normalizer using exact pairs from the original stream:
      - every computer_call_output has its matching computer_call (same call_id)
      - every computer_call is immediately preceded by its paired reasoning (if any)
      - ALL other reasoning items are dropped (no orphans possible)
    """
    if not request:
        return []

    callid_to_reasoning, _ = _pair_reasoning_calls(all_items)

    # 1) ensure each output has its call
    outs = [i for i in request if i.get("type") == "computer_call_output"]
    needed = {o.get("call_id") for o in outs if o.get("call_id")}
    have = {c.get("call_id") for c in request if c.get("type") == "computer_call"}

    if needed - have:
        missing = needed - have
        for it in all_items:
            if it.get("type") == "computer_call" and it.get("call_id") in missing:
                # place call before its first output occurrence
                first_out_idx = next(
                    (i for i, r in enumerate(request)
                     if r.get("type") == "computer_call_output" and r.get("call_id") == it.get("call_id")),
                    len(request)
                )
                request.insert(first_out_idx, it)

    # 2) insert EXACT paired reasoning immediately before each call (if any pair recorded)
    i = 0
    while i < len(request):
        it = request[i]
        if it.get("type") == "computer_call":
            cid = it.get("call_id")
            r = callid_to_reasoning.get(cid)
            if r:
                if r not in request:
                    request.insert(i, r)
                    i += 1
                else:
                    r_idx = request.index(r)
                    if r_idx != i - 1:
                        request.pop(r_idx)
                        request.insert(i, r)
                        i += 1
        i += 1

    # 3) DROP any reasoning not immediately followed by a computer_call in this request
    cleaned = []
    j = 0
    while j < len(request):
        it = request[j]
        if it.get("type") == "reasoning":
            nxt = request[j + 1] if j + 1 < len(request) else None
            if not (isinstance(nxt, dict) and nxt.get("type") == "computer_call"):
                j += 1
                continue  # drop orphan
        cleaned.append(it)
        j += 1

    return cleaned


def _force_include_reasoning_and_call(all_items: list[dict], request: list[dict], rs_id: str) -> list[dict]:
    """
    If the API names a specific missing reasoning (rs_id), force-include that
    reasoning and the next computer_call that follows it in the original stream,
    placing them before any matching call_output.
    """
    if not rs_id:
        return request

    rs_item = next((it for it in all_items if it.get("type") == "reasoning" and it.get("id") == rs_id), None)
    if not rs_item:
        return request

    try:
        rs_idx = all_items.index(rs_item)
    except ValueError:
        return request

    call_item = None
    for j in range(rs_idx + 1, len(all_items)):
        it = all_items[j]
        if it.get("type") == "computer_call":
            call_item = it
            break
    if not call_item:
        return request

    # remove any existing copies to re-order cleanly
    request = [it for it in request if it is not rs_item and it is not call_item]
    # insert pair just before its output if present
    first_out_idx = next(
        (i for i, it in enumerate(request)
         if it.get("type") == "computer_call_output" and it.get("call_id") == call_item.get("call_id")),
        len(request)
    )
    request[first_out_idx:first_out_idx] = [rs_item, call_item]
    return request


def _compact_items_for_next_request(
    all_items: list[dict],
    keep_images: int = 1,
    keep_tail: int = 30,
    keep_calls: int = 6,          # extra recent calls
    keep_reasons: int = 6,        # extra recent reasoning items (used as fillers if needed)
) -> list[dict]:
    """
    Compact request context while preserving CUA invariants:

      - Keep latest user 'message'.
      - Keep last `keep_tail` items as a base.
      - Keep only the latest `keep_images` input_image items.
      - Ensure each computer_call_output has its matching computer_call.
      - Ensure each computer_call is immediately preceded by its nearest prior 'reasoning' (if present in history).
      - Remove (or complete) any 'reasoning' that does not precede a computer_call in this request.
      - Optionally keep a few recent computer_call and reasoning items for resilience.
    """
    if not all_items:
        return []

    # --- start from a short tail & latest user message ---
    tail = list(all_items[-keep_tail:])

    last_user = next((it for it in reversed(all_items)
                      if it.get("type") == "message" and it.get("role") == "user"), None)
    if last_user and last_user not in tail:
        tail = [last_user] + tail

    # --- keep only the most recent K input_image(s) ---
    imgs = [i for i in tail if i.get("type") == "input_image"]
    if len(imgs) > keep_images:
        to_drop = set(id(i) for i in imgs[:-keep_images])
        tail = [i for i in tail if id(i) not in to_drop]

    # --- index helpers over full history ---
    idx_all = {id(it): i for i, it in enumerate(all_items)}

    def nearest_reasoning_before(idx_in_all: int) -> dict | None:
        for j in range(idx_in_all - 1, -1, -1):
            it = all_items[j]
            if it.get("type") == "reasoning":
                return it
            # (heuristic: you could early-stop at a user message)
        return None

    def nearest_call_after(idx_in_all: int) -> dict | None:
        for j in range(idx_in_all + 1, len(all_items)):
            it = all_items[j]
            if it.get("type") == "computer_call":
                return it
            # if you want, early-stop on next user message
        return None

    # --- 1) ensure each computer_call_output has its computer_call ---
    outputs = [i for i in tail if i.get("type") == "computer_call_output"]
    needed_call_ids = {o.get("call_id") for o in outputs if o.get("call_id")}

    tail_calls_by_id = {
        c.get("call_id"): idx
        for idx, c in enumerate(tail)
        if c.get("type") == "computer_call" and c.get("call_id")
    }

    for call_id in list(needed_call_ids):
        if call_id in tail_calls_by_id:
            continue
        call_item = None
        # scan backwards for speed (most recent first)
        for it in reversed(all_items):
            if it.get("type") == "computer_call" and it.get("call_id") == call_id:
                call_item = it
                break
        if call_item:
            first_out_idx = next(
                (i for i, it in enumerate(tail)
                 if it.get("type") == "computer_call_output" and it.get("call_id") == call_id),
                len(tail)
            )
            tail.insert(first_out_idx, call_item)

    # --- 2) ensure each computer_call has its preceding reasoning (if any) ---
    i = 0
    while i < len(tail):
        it = tail[i]
        if it.get("type") == "computer_call":
            has_prev_reason = (i - 1 >= 0 and tail[i - 1].get("type") == "reasoning")
            if not has_prev_reason:
                orig_idx = idx_all.get(id(it))
                if orig_idx is not None:
                    r = nearest_reasoning_before(orig_idx)
                    if r:
                        # insert reasoning just before the call if not already present
                        if r not in tail:
                            tail.insert(i, r)
                            i += 1  # skip over inserted reasoning
                        else:
                            # ensure ordering: move r to position i-1 if needed
                            r_idx = tail.index(r)
                            if r_idx != i - 1:
                                tail.pop(r_idx)
                                tail.insert(i, r)
                                i += 1
        i += 1

    # --- 3) drop or complete orphan reasonings (no following call in this request) ---
    # Build set of reasonings to keep: the nearest-before for each call we’re sending
    keep_reasoning_ids = set()
    for idx, it in enumerate(tail):
        if it.get("type") == "computer_call":
            orig_idx = idx_all.get(id(it))
            if orig_idx is not None:
                r = nearest_reasoning_before(orig_idx)
                if r:
                    keep_reasoning_ids.add(id(r))

    # Optionally, try to "complete" a dangling reasoning by inserting its nearest following call.
    # But to keep payload small, we only keep reasonings that serve a call; others are dropped.
    new_tail = []
    for it in tail:
        if it.get("type") == "reasoning":
            if id(it) not in keep_reasoning_ids:
                # drop orphan reasoning (prevents: "reasoning … provided without its required following item")
                continue
        new_tail.append(it)
    tail = new_tail

    # --- 4) resilience: keep a few recent calls and reasons (but don’t re-introduce orphans) ---
    recent_calls = [it for it in reversed(all_items) if it.get("type") == "computer_call"]
    for call in reversed(recent_calls[:keep_calls]):
        if call not in tail:
            # insert at end; its reasoning will be added by step (2) on next compaction
            tail.append(call)

    recent_reasons = [it for it in reversed(all_items) if it.get("type") == "reasoning"]
    # only add if it precedes a call we already carry (avoid creating new orphans)
    for r in reversed(recent_reasons[:keep_reasons]):
        if r in tail:
            continue
        r_idx_all = idx_all.get(id(r))
        if r_idx_all is None:
            continue
        following_call = nearest_call_after(r_idx_all)
        if following_call and following_call in tail:
            # place r just before that call (if not already right before)
            call_pos = tail.index(following_call)
            tail.insert(call_pos, r)

    return tail


class Agent:
    """
    A sample agent class that can be used to interact with a computer.

    Notes:
    - We KEEP 'reasoning' items in history because we're using the
      same model ('computer-use-preview') across turns. If you ever
      hand the conversation to a different model, strip 'reasoning'
      items at that boundary.
    """

    def __init__(
            self,
            model: str = "computer-use-preview",
            computer: Computer | None = None,
            tools: list[dict] | None = None,
            acknowledge_safety_check_callback: Callable = lambda *a, **k: True,
    ):
        self.model = model
        self.computer = computer
        self.tools = tools or []
        self.print_steps = True
        self.debug = True
        self.show_images = True
        self.acknowledge_safety_check_callback = acknowledge_safety_check_callback
        self._reason_queue: deque[str] = deque(maxlen=16)
        self.request_stop = False  # graceful Ctrl+C

        if computer:
            w, h = computer.get_dimensions()
            # Computer preview tool (required by CUA)
            self.tools += [{
                "type": "computer-preview",
                "display_width": w,
                "display_height": h,
                "environment": computer.get_environment(),
            }]

            # ---------- Function tools (selector-first navigation) ----------
            self.tools += [
                {
                    "type": "function",
                    "name": "wait_for_selector",
                    "description": "Wait until a CSS selector is visible & stable. Prefer before clicking.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector to wait for"},
                            "timeout_ms": {"type": "integer", "description": "Timeout in ms (default 4000)"}
                        },
                        "required": ["selector"]
                    }
                },
                {
                    "type": "function",
                    "name": "click_selector",
                    "description": "Click an element by CSS selector (center of element).",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "selector": {"type": "string"},
                            "timeout_ms": {"type": "integer", "description": "Timeout in ms (default 4000)"}
                        },
                        "required": ["selector"]
                    }
                },
                {
                    "type": "function",
                    "name": "click_text",
                    "description": "Click an element by visible text. Use exact=true for exact match.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "text": {"type": "string"},
                            "exact": {"type": "boolean", "description": "Exact match (default false)"},
                            "timeout_ms": {"type": "integer", "description": "Timeout in ms (default 4000)"}
                        },
                        "required": ["text"]
                    }
                },
                {
                    "type": "function",
                    "name": "type_selector",
                    "description": "Focus an input by selector and type text.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "selector": {"type": "string"},
                            "text": {"type": "string"},
                            "clear": {"type": "boolean", "description": "Clear first (default true)"},
                            "delay_ms": {"type": "integer", "description": "Key delay per char (default 0)"},
                            "timeout_ms": {"type": "integer", "description": "Timeout in ms (default 4000)"}
                        },
                        "required": ["selector", "text"]
                    }
                },
                {
                    "type": "function",
                    "name": "reload",
                    "description": "Reload the page reliably (use instead of keyboard shortcuts).",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {}
                    }
                },
            ]

    def debug_print(self, *args):
        if self.debug:
            pp(*args)

    @staticmethod
    def _extract_reasoning_text(item: dict) -> str:
        """
        Extract a readable one-liner from a reasoning item if present.
        Falls back to '(no reasoning provided)' instead of printing raw IDs.
        """
        # 1) Docs shape: {"summary":[{"type":"summary_text","text":"..."}]}
        summary = item.get("summary")
        if isinstance(summary, list):
            for s in summary:
                if isinstance(s, dict) and s.get("type") in ("summary_text", "text"):
                    t = s.get("text")
                    if isinstance(t, str) and t.strip():
                        return t.strip()
        elif isinstance(summary, str) and summary.strip():
            return summary.strip()

        # 2) General deep search for a usable text field
        def _deep_first_text(o):
            PREF = ("summary_text", "text", "message", "rationale", "explanation", "why", "reason")
            if isinstance(o, dict):
                for k in PREF:
                    v = o.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                for v in o.values():
                    r = _deep_first_text(v)
                    if r:
                        return r
            elif isinstance(o, (list, tuple)):
                for v in o:
                    r = _deep_first_text(v)
                    if r:
                        return r
            elif isinstance(o, str) and o.strip():
                return o.strip()
            return None

        text = _deep_first_text(item.get("content")) or _deep_first_text(item)
        if text:
            return text if len(text) <= 200 else (text[:197] + "…")

        # 3) Compact JSON peek of known fields (short & informative only)
        try:
            payload = {k: item.get(k) for k in ("summary", "content") if k in item}
            preview = json.dumps(payload, ensure_ascii=False)
            if preview and preview != "{}" and len(preview) <= 160:
                return preview
        except Exception:
            pass

        # 4) Final fallback — do NOT print the raw rs_… id
        return "(no reasoning provided)"

    def _push_reasoning(self, item: dict) -> None:
        reason = (self._extract_reasoning_text(item) or "").strip()

        # Optional: avoid printing noisy placeholders
        if reason.lower() in {"", "(no reasoning provided)"}:
            # still queue an empty marker if you want the next action comment to omit
            self._reason_queue.append("")
            return

        # Debounce overlay updates (set only if changed)
        if hasattr(self.computer, "set_overlay_text"):
            try:
                # Truncate to keep the bubble readable (adjust length as you like)
                short = reason if len(reason) <= 280 else (reason[:277] + "…")
                # If your BasePlaywrightComputer tracks last text, you can skip if unchanged.
                self.computer.set_overlay_text(short)
            except Exception:
                pass

        # Optional: print steps if you want console visibility
        if self.print_steps:
            print(f"[Reasoning] {reason}")

        self._reason_queue.append(reason)

    def _pop_reason_for_action(self) -> str:
        return self._reason_queue.popleft() if self._reason_queue else ""

    def handle_item(self, item):
        """Handle each item; may cause a computer action + screenshot."""
        itype = item.get("type")

        if itype == "message":
            if self.print_steps:
                content = item.get("content") or []
                text = None
                if isinstance(content, list) and content:
                    c0 = content[0]
                    if isinstance(c0, dict):
                        text = c0.get("text")
                print(f"Message: {text or '(no text)'}")
            return []

        if itype == "function_call":
            name, raw_args = item["name"], json.loads(item.get("arguments") or "{}")
            if self.print_steps:
                print(f"{name}({raw_args})")

            status = "skipped"
            if hasattr(self.computer, name):
                method = getattr(self.computer, name)
                try:
                    # only pass parameters the method actually accepts
                    sig = inspect.signature(method)
                    kwargs = {k: v for k, v in raw_args.items() if k in sig.parameters}
                    method(**kwargs)
                    status = "success"
                except Exception as e:
                    status = f"error: {e!r}"

            return [{
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": status,
            }]

        if itype == "reasoning":
            self._push_reasoning(item)
            return []

        if itype == "computer_call":
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}

            reason = self._pop_reason_for_action()
            suffix = f"  # {reason}" if reason else ""
            if self.print_steps:
                print(f"Computer call: {action_type}({action_args}){suffix}")

            # Execute the action
            getattr(self.computer, action_type)(**action_args)

            # Screenshot (PNG base64 for stitched MP4); preview if requested
            screenshot_base64 = self.computer.screenshot()
            if self.show_images:
                show_image(screenshot_base64)

            # Build the vision payload once
            use_inline = os.getenv("VISION_INLINE", "0").lower() in ("1", "true", "yes")
            img_payload: dict
            file_id: str | None = None  # define up-front so it's always bound

            try:
                # Compress PNG → JPEG (smaller for either inline or file upload)
                jpeg_bytes = compress_b64_png_to_jpeg_bytes(
                    screenshot_base64, max_w=1280, max_h=960, quality=70
                )

                if use_inline:
                    # Inline JPEG data URL (no Files API)
                    import base64
                    jpeg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                    img_payload = {"type": "input_image", "image_url": f"data:image/jpeg;base64,{jpeg_b64}"}
                    if self.debug:
                        print(f"[Diag] vision=inline, jpeg={len(jpeg_bytes)} bytes")
                else:
                    # Upload JPEG and use file_id
                    file_id = upload_image_get_file_id(jpeg_bytes)
                    if file_id:
                        img_payload = {"type": "input_image", "file_id": file_id}
                        if self.debug:
                            print(f"[Diag] vision=file_id, jpeg={len(jpeg_bytes)} bytes, file_id={file_id}")
                    else:
                        # Fallback to inline PNG if upload failed
                        img_payload = {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_base64}"}
                        if self.debug:
                            print("[Diag] upload failed; falling back to inline png")
            except Exception as e:
                # Last-resort fallback: inline PNG
                img_payload = {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_base64}"}
                if self.debug:
                    print(f"[Diag] vision fallback to png inline due to: {e!r}")

            # (optional) manual safety acks left disabled per your choice
            pending_checks = item.get("pending_safety_checks", [])

            # Return computer_call_output with the built image payload
            call_output = {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "acknowledged_safety_checks": pending_checks,
                "output": img_payload,
            }

            # Browser extra: include current URL for guardrails/diagnostics
            if self.computer.get_environment() == "browser":
                current_url = self.computer.get_current_url()
                check_blocklisted_url(current_url)
                call_output["output"]["current_url"] = current_url

            return [call_output]

        return []

    def run_full_turn(self, input_items, print_steps=True, debug=True, show_images=True):
        self.print_steps = print_steps
        self.debug = debug
        self.show_images = show_images
        new_items = []

        while new_items[-1].get("role") != "assistant" if new_items else True:
            if self.request_stop:
                if self.print_steps:
                    print("[Exit] Stop requested — leaving turn loop.")
                break

            # 1) compact, then enforce strict pair invariants
            next_input = _compact_base(input_items + new_items, keep_images=1, keep_tail=30)
            next_input = _enforce_cua_via_pairs(input_items + new_items, next_input)

            # 2) send with up to 3 targeted retries
            max_retries = 3
            attempt = 0
            while True:
                if self.debug:
                    print("[Sending]",
                          [(it.get("type"), it.get("id") or it.get("call_id")) for it in next_input])

                try:
                    response = create_response(
                        model=self.model,
                        input=next_input,
                        tools=self.tools,
                        truncation="auto",
                    )
                    break  # success
                except RuntimeError as e:
                    msg = str(e)

                    # A) API named a specific missing reasoning
                    m = re.search(r"required 'reasoning' item:\s*'(rs_[A-Za-z0-9_]+)'", msg)
                    if m and attempt < max_retries:
                        rs_id = m.group(1)
                        widened = _compact_base(input_items + new_items, keep_images=1, keep_tail=80)
                        widened = _force_include_reasoning_and_call(input_items + new_items, widened, rs_id)
                        next_input = _enforce_cua_via_pairs(input_items + new_items, widened)
                        attempt += 1
                        continue

                    # B) classic: output without call
                    if "No tool call found for computer call with call_id" in msg and attempt < max_retries:
                        widened = _compact_base(input_items + new_items, keep_images=1, keep_tail=80)
                        next_input = _enforce_cua_via_pairs(input_items + new_items, widened)
                        attempt += 1
                        continue

                    # C) give up gracefully
                    print(f"[Exit] API error: {msg}")
                    return new_items  # end turn, no crash

            if "output" not in response:
                print(f"[Exit] Unexpected response shape (no 'output').")
                break

            out = response["output"]
            new_items += out

            for item in out:
                if self.request_stop:
                    if self.print_steps:
                        print("[Exit] Stop requested — finishing after current items.")
                    break
                new_items += self.handle_item(item)

        return new_items
