import os
import requests
from dotenv import load_dotenv
import json
import base64
from PIL import Image
from io import BytesIO
import io
from urllib.parse import urlparse
from typing import List, Dict, Any

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
    """Return a copy of the message with large image payloads omitted.

    This is useful for debug logging so that base64-encoded screenshots don't
    overwhelm the console. The structure of the message is preserved, only the
    image data is replaced with a placeholder string.
    """

    if msg.get("type") == "computer_call_output":
        output = msg.get("output", {})
        if isinstance(output, dict):
            sanitized = msg.copy()
            new_output = dict(output)
            if "image_url" in new_output:
                new_output["image_url"] = "[omitted]"
            if "image_base64" in new_output:
                new_output["image_base64"] = "[omitted]"
            sanitized["output"] = new_output
            return sanitized
    return msg


# Output-only item types that must never be sent back in the next request.
MODEL_ONLY_TYPES = {"reasoning", "output_text"}


def strip_model_only_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove model-only items so we never send them back in `input`."""
    clean: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        if t in MODEL_ONLY_TYPES:
            continue
        clean.append(it)
    return clean


def coerce_input_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare messages for the API request.

    Drops non-dict items, filters out model-only item types and otherwise keeps
    the messages unchanged so that fields like image_base64 remain intact for
    the model to consume.
    """

    out: List[Dict[str, Any]] = []
    for it in items:
        if it is None or not isinstance(it, dict):
            continue
        out.append(it)
    return strip_model_only_items(out)


def is_error_response(resp: dict) -> bool:
    return isinstance(resp, dict) and "output" not in resp and "error" in resp


def summarize_error(resp: dict) -> str:
    err = (resp or {}).get("error") or {}
    msg = err.get("message") or str(err)
    typ = err.get("type")
    code = err.get("code")
    return f"{typ or 'error'}: {msg}" + (f" (code={code})" if code else "")

def create_response(**kwargs):
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    openai_org = os.getenv("OPENAI_ORG")
    if openai_org:
        headers["Openai-Organization"] = openai_org

    response = requests.post(url, headers=headers, json=kwargs)

    if response.status_code != 200:
        print(f"Error: {response.status_code} {response.text}")

    return response.json()


def check_blocklisted_url(url: str) -> None:
    """Raise ValueError if the given URL (including subdomains) is in the blocklist."""
    hostname = urlparse(url).hostname or ""
    if any(
        hostname == blocked or hostname.endswith(f".{blocked}")
        for blocked in BLOCKED_DOMAINS
    ):
        raise ValueError(f"Blocked URL: {url}")
