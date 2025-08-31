from computers import Computer
from utils import (
    create_response,
    show_image,
    pp,
    sanitize_message,
    check_blocklisted_url,
    coerce_input_items,
    is_error_response,
    summarize_error,
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
        self.debug = False
        self.show_images = False
        self.acknowledge_safety_check_callback = acknowledge_safety_check_callback

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

    def handle_item(self, item):
        """Handle each item; may cause a computer action + screenshot."""
        if item["type"] == "message":
            if self.print_steps:
                print(item["content"][0]["text"])

        if item["type"] == "function_call":
            name, args = item["name"], json.loads(item["arguments"])
            if self.print_steps:
                print(f"{name}({args})")

            if hasattr(self.computer, name):  # if function exists on computer, call it
                method = getattr(self.computer, name)
                result = method(**args)

                if name == "screenshot" and isinstance(result, dict):
                    if self.show_images:
                        url = result.get("image_url", "")
                        if url.startswith("data:image"):
                            show_image(url.split(",", 1)[1])

                    output = {}
                    if "image_url" in result:
                        output["image_url"] = result["image_url"]
                    if "file_id" in result:
                        output["file_id"] = result["file_id"]

                    return [
                        {
                            "type": "function_call_output",
                            "call_id": item["call_id"],
                            "output": output,
                        }
                    ]

            return [
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": "success",  # hard-coded output for demo
                }
            ]

        if item["type"] == "computer_call":
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}
            if self.print_steps:
                print(f"{action_type}({action_args})")

            method = getattr(self.computer, action_type)
            method(**action_args)

            screenshot_item = self.computer.screenshot()
            if self.show_images:
                url = screenshot_item.get("image_url", "")
                if url.startswith("data:image"):
                    show_image(url.split(",", 1)[1])

            # if user doesn't ack all safety checks exit with error
            pending_checks = item.get("pending_safety_checks", [])
            for check in pending_checks:
                message = check["message"]
                if not self.acknowledge_safety_check_callback(message):
                    raise ValueError(
                        f"Safety check failed: {message}. Cannot continue with unacknowledged safety checks."
                    )

            call_output = {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "acknowledged_safety_checks": pending_checks,
                "output": {"screenshot": True},
            }

            # additional URL safety checks for browser environments
            if self.computer.get_environment() == "browser":
                current_url = self.computer.get_current_url()
                check_blocklisted_url(current_url)
                call_output["output"]["current_url"] = current_url

            return [call_output, screenshot_item]
        return []

    def run_full_turn(
        self, input_items, print_steps=True, debug=False, show_images=False
    ):
        self.print_steps = print_steps
        self.debug = debug
        self.show_images = show_images
        new_items = []

        # keep looping until we get a final response
        while new_items[-1].get("role") != "assistant" if new_items else True:
            sanitized = coerce_input_items(input_items + new_items)
            if self.debug:
                print("[Sending items]")
                for i, it in enumerate(sanitized):
                    t = it.get("type")
                    keys = list(it.keys())
                    print(i, t, "keys=", keys)
            self.debug_print([sanitize_message(msg) for msg in sanitized])

            try:
                response = create_response(
                    model=self.model,
                    input=sanitized,
                    tools=self.tools,
                    truncation="auto",
                )
            except Exception as e:
                if self.debug:
                    print("API call failed:", repr(e))
                raise

            self.debug_print(response)

            if is_error_response(response):
                msg = summarize_error(response)
                raise ValueError(f"Model response error: {msg}")

            if "output" not in response:
                if self.debug:
                    print("Unexpected response payload:", response)
                raise ValueError("No output from model")

            response_items = response["output"]
            new_items += response_items
            for item in response_items:
                new_items += self.handle_item(item)

        return new_items
