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
    if msg.get("type") == "input_image":
        sanitized = msg.copy()
        if "image_url" in sanitized:
            sanitized["image_url"] = "[omitted]"
        if "image_base64" in sanitized:
            sanitized["image_base64"] = "[omitted]"
        return sanitized
    return msg


# Output-only item types that must never be sent back in the next request.
_MODEL_ONLY = {"reasoning", "output_text"}


def strip_model_only_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove model-only items so we never send them back in `input`."""
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") in _MODEL_ONLY:
            continue
        out.append(it)
    return out


def normalize_image_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure every input_image has exactly one of image_url or file_id.
    Drop malformed ones; fix common 'output' wrapping if possible.
    """
    fixed: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") != "input_image":
            fixed.append(it)
            continue

        if "image_url" in it or "file_id" in it:
            fixed.append(it)
            continue

        out = it.get("output")
        if isinstance(out, dict) and ("image_url" in out or "file_id" in out):
            if "image_url" in out:
                fixed.append({"type": "input_image", "image_url": out["image_url"]})
            else:
                fixed.append({"type": "input_image", "file_id": out["file_id"]})
            continue

        # drop malformed image
        # optionally, could log here

    return fixed


def coerce_input_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base = [it for it in items if isinstance(it, dict)]
    base = strip_model_only_items(base)
    base = normalize_image_items(base)
    return base


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
