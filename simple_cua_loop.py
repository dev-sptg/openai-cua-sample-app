from collections import deque

from computers import Computer
from computers import LocalPlaywrightComputer
from utils import create_response, check_blocklisted_url

# Buffer of reasoning summaries to attach to the next action(s)
_reason_queue = deque(maxlen=8)


def acknowledge_safety_check_callback(message: str) -> bool:
    response = input(
        f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): "
    ).lower()
    return response.strip() == "y"


def _print_reasoning(item: dict) -> None:
    """Pretty-print the model's reasoning summary (if present)."""
    # item looks like:
    # {"type":"reasoning","id":"rs_...","summary":[{"type":"summary_text","text":"..."}], ...}
    summary = item.get("summary") or []
    texts = [s.get("text") for s in summary if isinstance(s, dict) and s.get("type") == "summary_text"]
    if texts:
        print(f"Reasoning: {texts[0]}")
    else:
        # fallback: if future structures change
        print(f":Reasoning: {item.get('id','(no-id)')}")


def _push_reasoning(item: dict) -> None:
    """Extract and buffer the model's reasoning summary for later printing."""
    summary = item.get("summary") or []
    texts = [
        s.get("text")
        for s in summary
        if isinstance(s, dict) and s.get("type") == "summary_text" and s.get("text")
    ]
    if texts:
        reason = texts[0].strip()
        print(f"[Reasoning] {reason}")
        _reason_queue.append(reason)
    else:
        # fallback (future-proof): still expose there *was* reasoning
        rid = item.get("id", "(no-id)")
        print(f"[Reasoning] {rid}")
        _reason_queue.append(f"(reasoning id {rid})")


def _pop_reason_for_action() -> str:
    """Return the most recent reasoning (if any) to annotate the next action."""
    if _reason_queue:
        return _reason_queue.popleft()
    return ""


def handle_item(item, computer: Computer):
    """Handle each item; may cause a computer action + screenshot."""
    if item["type"] == "message":  # print messages
        print(item["content"][0]["text"])

    if item["type"] == "reasoning":
        # Print the model's short plan/intent for the next action
        _push_reasoning(item)
        return []

    if item["type"] == "computer_call":  # perform computer actions
        action = item["action"]
        action_type = action["type"]
        action_args = {k: v for k, v in action.items() if k != "type"}

        # annotate the action with the most recent reasoning (if any)
        reason = _pop_reason_for_action()
        suffix = f"  # {reason}" if reason else ""
        print(f"{action_type}({action_args}){suffix}")

        # give our computer environment action to perform
        getattr(computer, action_type)(**action_args)

        screenshot_base64 = computer.screenshot()

        pending_checks = item.get("pending_safety_checks", [])
        for check in pending_checks:
            if not acknowledge_safety_check_callback(check["message"]):
                raise ValueError(f"Safety check failed: {check['message']}")

        # return value informs model of the latest screenshot
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
        if computer.get_environment() == "browser":
            current_url = computer.get_current_url()
            call_output["output"]["current_url"] = current_url
            check_blocklisted_url(current_url)

        return [call_output]

    return []


def main():
    """Run the CUA (Computer Use Assistant) loop, using Local Playwright."""
    with LocalPlaywrightComputer() as computer:
        dimensions = computer.get_dimensions()
        tools = [
            {
                "type": "computer-preview",
                "display_width": dimensions[0],
                "display_height": dimensions[1],
                "environment": computer.get_environment(),
            }
        ]

        items = []
        while True:  # get user input forever
            user_input = input("> ")
            items.append({"role": "user", "content": user_input})

            while True:  # keep looping until we get a final response
                response = create_response(
                    model="computer-use-preview",
                    input=items,
                    tools=tools,
                    truncation="auto",
                )

                if "output" not in response:
                    print(f"Response: {response}")
                    raise ValueError("No output from model")

                items += response["output"]

                for item in response["output"]:
                    items += handle_item(item, computer)

                if items[-1].get("role") == "assistant":
                    break


if __name__ == "__main__":
    main()
