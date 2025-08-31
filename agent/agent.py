from collections import deque
import json
from typing import Callable

from computers import Computer
from utils import (
    create_response,
    show_image,
    pp,
    sanitize_message,
    check_blocklisted_url,
)


def _compact_items_for_next_request(
    all_items: list[dict],
    keep_images: int = 1,
    keep_tail: int = 30,
    keep_calls: int = 6,          # keep a few recent computer_call items just in case
) -> list[dict]:
    """
    Build a compact request context while preserving protocol requirements:
      - keep latest user message,
      - keep last `keep_tail` items,
      - keep last `keep_images` input_image items,
      - ensure each computer_call_output has its matching computer_call present,
      - keep last `keep_calls` computer_call items as extra safety.
    """
    if not all_items:
        return []

    # Start from a small tail
    tail = list(all_items[-keep_tail:])

    # Ensure latest user instruction is included
    last_user = next((it for it in reversed(all_items) if it.get("type") == "message" and it.get("role") == "user"), None)
    if last_user and last_user not in tail:
        tail = [last_user] + tail

    # Keep only the most recent K input_image(s)
    imgs = [i for i in tail if i.get("type") == "input_image"]
    if len(imgs) > keep_images:
        to_drop = set(id(i) for i in imgs[:-keep_images])
        tail = [i for i in tail if id(i) not in to_drop]

    # Collect required call_ids from any computer_call_output in the tail
    outputs = [i for i in tail if i.get("type") == "computer_call_output"]
    needed_call_ids = {o.get("call_id") for o in outputs if o.get("call_id")}

    # Map existing calls present in the tail
    tail_calls_by_id = {
        c.get("call_id"): idx
        for idx, c in enumerate(tail)
        if c.get("type") == "computer_call" and c.get("call_id")
    }

    # For each needed call_id, ensure its computer_call is in the tail.
    # If missing, scan backwards in all_items and pull it in, placing it just before the first matching output.
    for call_id in list(needed_call_ids):
        if call_id in tail_calls_by_id:
            continue
        # find the original computer_call in full history (walk backward for speed)
        call_item = None
        for it in reversed(all_items):
            if it.get("type") == "computer_call" and it.get("call_id") == call_id:
                call_item = it
                break
        if call_item:
            # insert the call just before its first output occurrence in tail
            first_out_idx = next(
                (i for i, it in enumerate(tail) if it.get("type") == "computer_call_output" and it.get("call_id") == call_id),
                len(tail)
            )
            tail.insert(first_out_idx, call_item)

    # As extra safety, keep a few recent computer_call items even if not strictly required
    recent_calls = [it for it in reversed(all_items) if it.get("type") == "computer_call"]
    for call in reversed(recent_calls[:keep_calls]):
        if call not in tail:
            tail.append(call)

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
        model="computer-use-preview",
        computer: Computer = None,
        tools: list[dict] = [],
        acknowledge_safety_check_callback: Callable = lambda *a, **k: True,
    ):
        self.model = model
        self.computer = computer
        self.tools = tools
        self.print_steps = True
        self.debug = True
        self.show_images = True
        self.acknowledge_safety_check_callback = acknowledge_safety_check_callback

        # buffer to print "why" next to the next action
        self._reason_queue: deque[str] = deque(maxlen=16)

        if computer:
            dimensions = computer.get_dimensions()
            self.tools += [
                {
                    "type": "computer-preview",
                    "display_width": dimensions[0],
                    "display_height": dimensions[1],
                    "environment": computer.get_environment(),
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
        reason = self._extract_reasoning_text(item)
        '''
        if self.print_steps:
            print(f"[Reasoning] {reason}")
        '''
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
            name, args = item["name"], json.loads(item.get("arguments") or "{}")
            if self.print_steps:
                print(f"{name}({args})")
            if hasattr(self.computer, name):
                getattr(self.computer, name)(**args)
            return [
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": "success",
                }
            ]

        if itype == "reasoning":
            # Print and queue the reasoning, then continue to wait for the call
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

            # execute the action
            getattr(self.computer, action_type)(**action_args)

            # screenshot after the action
            screenshot_base64 = self.computer.screenshot()
            if self.show_images:
                show_image(screenshot_base64)

            # (optional) manual safety acks - left disabled per your choice
            pending_checks = item.get("pending_safety_checks", [])

            # return computer_call_output with a valid input_image
            call_output = {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "acknowledged_safety_checks": pending_checks,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                },
            }

            # extra URL safety in browser envs
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

        # loop until final assistant message
        while new_items[-1].get("role") != "assistant" if new_items else True:
            # compact to avoid huge requests (keeps latest screenshot)
            next_input = _compact_items_for_next_request(input_items + new_items, keep_images=1, keep_tail=30)

            self.debug_print([sanitize_message(msg) for msg in next_input])

            response = create_response(
                model=self.model,
                input=next_input,     # send compacted history
                tools=self.tools,
                truncation="auto",
            )
            self.debug_print(f"Debug-Response: {response}")

            if "output" not in response:
                print(f"Response: {response}")
                raise ValueError("No output from model")

            out = response["output"]
            new_items += out

            for item in out:
                new_items += self.handle_item(item)

        return new_items
