from collections import deque

from computers import Computer
from utils import (
    create_response,
    show_image,
    pp,
    sanitize_message,
    check_blocklisted_url,
)
import json
from typing import Callable


class Agent:
    """
    A sample agent class that can be used to interact with a computer.

    (See simple_cua_loop.py for a simple example without an agent.)
    """

    def __init__(
        self,
        model="computer-use-preview",
        computer: Computer = None,
        tools: list[dict] = [],
        acknowledge_safety_check_callback: Callable = lambda: False,
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

    def _push_reasoning(self, item: dict) -> None:
        """Extract & buffer reasoning summary, print it."""
        summary = item.get("summary") or []
        texts = [s.get("text") for s in summary
                 if isinstance(s, dict) and s.get("type") == "summary_text" and s.get("text")]
        reason = (texts[0].strip() if texts else f"(reasoning id {item.get('id','?')})")
        if self.print_steps:
            print(f"[Reasoning] {reason}")
        self._reason_queue.append(reason)

    def _pop_reason_for_action(self) -> str:
        return self._reason_queue.popleft() if self._reason_queue else ""

    def handle_item(self, item):
        """Handle each item; may cause a computer action + screenshot."""
        if item["type"] == "message":
            if self.print_steps:
                print(f'Message: {item["content"][0]["text"]}')

        if item["type"] == "function_call":
            name, args = item["name"], json.loads(item["arguments"])
            if self.print_steps:
                print(f"{name}({args})")

            if hasattr(self.computer, name):  # if function exists on computer, call it
                method = getattr(self.computer, name)
                method(**args)
            return [
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": "success",  # hard-coded output for demo
                }
            ]

        if item["type"] == "reasoning":
            # Print and queue the reasoning, then continue to wait for the call
            self._push_reasoning(item)
            return []

        if item["type"] == "computer_call":
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}

            reason = self._pop_reason_for_action()
            suffix = f"  # {reason}" if reason else ""
            if self.print_steps:
                print(f"Computer call: {action_type}({action_args}){suffix}")

            method = getattr(self.computer, action_type)
            method(**action_args)

            screenshot_base64 = self.computer.screenshot()
            if self.show_images:
                show_image(screenshot_base64)

            # if user doesn't ack all safety checks exit with error
            pending_checks = item.get("pending_safety_checks", [])
            '''
            for check in pending_checks:
                message = check["message"]
                if not self.acknowledge_safety_check_callback(message):
                    raise ValueError(
                        f"Safety check failed: {message}. Cannot continue with unacknowledged safety checks."
                    )
            '''
            call_output = {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "acknowledged_safety_checks": pending_checks,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                },
            }

            # additional URL safety checks for browser environments
            if self.computer.get_environment() == "browser":
                current_url = self.computer.get_current_url()
                check_blocklisted_url(current_url)
                call_output["output"]["current_url"] = current_url

            return [call_output]
        return []

    def run_full_turn(
        self, input_items, print_steps=True, debug=True, show_images=True
    ):
        self.print_steps = print_steps
        self.debug = debug
        self.show_images = show_images
        new_items = []

        # keep looping until we get a final response
        while new_items[-1].get("role") != "assistant" if new_items else True:
            self.debug_print([sanitize_message(msg) for msg in input_items + new_items])

            response = create_response(
                model=self.model,
                input=input_items + new_items,
                tools=self.tools,
                truncation="auto",
            )
            self.debug_print(f"Debug-Response: {response}")

            if "output" not in response and self.debug:
                print(f"Response: {response}")
                raise ValueError("No output from model")
            else:
                new_items += response["output"]
                for item in response["output"]:
                    new_items += self.handle_item(item)

        return new_items
