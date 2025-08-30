import argparse
import os
import pathlib
from agent.agent import Agent
from computers.config import *
from computers.default import *
from computers import computers_config


def acknowledge_safety_check_callback(message: str) -> bool:
    response = input(
        f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): "
    ).lower()
    return response.lower().strip() == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Select a computer environment from the available options."
    )
    parser.add_argument(
        "--computer",
        choices=computers_config.keys(),
        help="Choose the computer environment to use.",
        default="local-playwright",
    )
    parser.add_argument(
        "--input",
        help="Initial instruction text (one-shot)",
    )
    parser.add_argument(
        "--task-file",
        help="Path to a file with the initial instruction",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress instruction preview",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show images during the execution.",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        help="Start the browsing session with a specific URL (only for browser environments).",
        default="https://chatbot-2hld.onrender.com/sessions",
    )
    args = parser.parse_args()

    instruction = None

    if args.input:
        instruction = args.input
    elif args.task_file:
        p = pathlib.Path(args.task_file)
        instruction = p.read_text(encoding="utf-8")
    else:
        instruction = os.getenv("DEFAULT_INSTRUCTION")

    if not instruction:
        instruction = input("> ")

    if not args.quiet:
        preview = instruction.strip().replace("\n", " ")
        print(f"[Instruction] {preview[:120]}{'…' if len(preview) > 120 else ''}")

    ComputerClass = computers_config[args.computer]

    with ComputerClass() as computer:
        agent = Agent(
            computer=computer,
            acknowledge_safety_check_callback=acknowledge_safety_check_callback,
        )
        items = [{"role": "user", "content": instruction}]

        if args.computer in ["browserbase", "local-playwright"]:
            if not args.start_url.startswith("http"):
                args.start_url = "https://" + args.start_url
            agent.computer.goto(args.start_url)

        output_items = agent.run_full_turn(
            items,
            print_steps=True,
            show_images=args.show,
            debug=args.debug,
        )
        items += output_items


if __name__ == "__main__":
    main()
