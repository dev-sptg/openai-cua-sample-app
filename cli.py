import argparse
import signal
import sys
from agent.agent import Agent
from computers.config import *
from computers.default import *
from computers import computers_config

# keep a module-level reference so the handler can flip it
agent = None


def acknowledge_safety_check_callback(message: str) -> bool:
    resp = input(
        f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): "
    ).lower()
    return resp.strip() == "y"


def _sigint_handler(signum, frame):
    # 1) ask the agent to stop after the current action
    if agent:
        agent.request_stop = True
    # 2) also break out of any blocking input() immediately
    #    by raising KeyboardInterrupt for this turn
    raise KeyboardInterrupt


def main():
    global agent
    parser = argparse.ArgumentParser(
        description="Select a computer environment from the available options."
    )
    parser.add_argument("--computer", choices=computers_config.keys(),
                        default="local-playwright")
    parser.add_argument("--input", type=str, default=None,
                        help="Initial instruction (skip console prompt).")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--start-url", type=str,
                        default="https://chatbot-2hld.onrender.com/sessions")
    args = parser.parse_args()

    # install handler
    signal.signal(signal.SIGINT, _sigint_handler)

    ComputerClass = computers_config[args.computer]

    try:
        with ComputerClass() as computer:
            agent = Agent(
                computer=computer,
                acknowledge_safety_check_callback=acknowledge_safety_check_callback,
            )
            items: list[dict] = []

            if args.computer in ["browserbase", "local-playwright"]:
                if not args.start_url.startswith("http"):
                    args.start_url = "https://" + args.start_url
                agent.computer.goto(args.start_url)
                try:
                    if hasattr(agent.computer, "ensure_overlay"):
                        agent.computer.ensure_overlay()
                        print("[Overlay] ensure_overlay() called")
                except Exception:
                    pass

            while True:
                # If a previous Ctrl+C requested stop, don’t prompt again
                if agent.request_stop:
                    print("[Exit] Stop requested — shutting down…")
                    break

                try:
                    user_text = args.input or input("> ")
                except KeyboardInterrupt:
                    # Ctrl+C during input(): stop immediately
                    print("\n[Exit] Ctrl+C — exiting.")
                    break
                except EOFError as e:
                    print(f"[Exit] Stdin closed: {e}")
                    break

                if user_text.strip().lower() in {"exit", "quit", ":q"}:
                    print("[Exit] User requested exit.")
                    break

                # Responses API message shape
                items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                })

                try:
                    output_items = agent.run_full_turn(
                        items, print_steps=True, show_images=args.show, debug=args.debug
                    )
                except KeyboardInterrupt:
                    # Ctrl+C while the agent is mid-turn: we already set request_stop in the handler
                    print("\n[Exit] Interrupted — finishing shutdown.")
                    break

                items += output_items
                args.input = None

                # If a Ctrl+C happened during the turn, leave gracefully
                if agent.request_stop:
                    print("[Exit] Stop requested — done.")
                    break

    finally:
        # If the handler didn’t run, restore default to avoid duplicate prints on process exit
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("[Exit] Done.")
        # Optional: sys.exit(0)


if __name__ == "__main__":
    main()
