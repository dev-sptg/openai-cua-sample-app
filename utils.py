import os
import requests
from dotenv import load_dotenv
import json
import base64
from PIL import Image
from io import BytesIO
import io
from urllib.parse import urlparse

load_dotenv(override=True)

BLOCKED_DOMAINS = [
    "maliciousbook.com",
    "evilvideos.com",
    "darkwebforum.com",
    "shadytok.com",
    "suspiciouspins.com",
    "ilanbigio.com",
]


def pp(obj):
    print(json.dumps(obj, indent=4))


def show_image(base_64_image):
    image_data = base64.b64decode(base_64_image)
    image = Image.open(BytesIO(image_data))
    image.show()


def calculate_image_dimensions(base_64_image):
    image_data = base64.b64decode(base_64_image)
    image = Image.open(io.BytesIO(image_data))
    return image.size


def sanitize_message(msg: dict) -> dict:
    """Return a copy of the message with image_url omitted for computer_call_output messages."""
    if msg.get("type") == "computer_call_output":
        output = msg.get("output", {})
        if isinstance(output, dict):
            sanitized = msg.copy()
            sanitized["output"] = {**output, "image_url": "[omitted]"}
            return sanitized
    return msg


def create_response(**kwargs):
    """
    Thin wrapper around Responses API with:
      - reasoning summaries enabled (summary='auto')
      - defensive JSON handling for non-JSON upstream errors
    """
    url = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    openai_org = os.getenv("OPENAI_ORG")
    if openai_org:
        headers["OpenAI-Organization"] = openai_org  # correct casing

    # Opt-in to reasoning summaries; allow env override
    reasoning_effort = os.getenv("REASONING_EFFORT", "medium")
    reasoning_summary = os.getenv("REASONING_SUMMARY", "auto")
    payload = {
        **kwargs,
        "reasoning": {"effort": reasoning_effort, "summary": reasoning_summary},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    # Non-JSON responses (e.g., proxies returning HTML) → readable error
    try:
        data = resp.json()
    except Exception:
        text = (resp.text or "")[:800]
        raise RuntimeError(f"API {resp.status_code} returned non-JSON response:\n{text}")

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {json.dumps(data)[:800]}")

    return data


def check_blocklisted_url(url: str) -> None:
    """Raise ValueError if the given URL (including subdomains) is in the blocklist."""
    hostname = urlparse(url).hostname or ""
    if any(
        hostname == blocked or hostname.endswith(f".{blocked}")
        for blocked in BLOCKED_DOMAINS
    ):
        raise ValueError(f"Blocked URL: {url}")
