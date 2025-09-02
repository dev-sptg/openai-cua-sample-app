from dotenv import load_dotenv
import io
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import base64
import os, json, time, random, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv(override=True)

BLOCKED_DOMAINS = [
    "maliciousbook.com",
    "evilvideos.com",
    "darkwebforum.com",
    "shadytok.com",
    "suspiciouspins.com",
    "ilanbigio.com",
]


def _make_session() -> requests.Session:
    """A session with sane retry defaults for transient network issues."""
    s = requests.Session()
    retry = Retry(
        total=0,                  # we do our own logical retries below
        connect=0, read=0,        # adapter-level retries off (to keep control)
        backoff_factor=0,         # no adapter backoff
        status_forcelist=[],      # no status retries here
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_SESSION = _make_session()

def compress_b64_png_to_jpeg_bytes(b64_png: str, max_w: int = 1280, max_h: int = 960, quality: int = 70) -> bytes:
    """
    Decode a base64 PNG screenshot, downscale, and re-encode as JPEG bytes.
    Keeps aspect ratio; good default balance for vision and request size.
    """
    raw = base64.b64decode(b64_png)
    im = Image.open(BytesIO(raw)).convert("RGB")
    im.thumbnail((max_w, max_h))  # in-place, keeps aspect ratio
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def upload_image_get_file_id(image_bytes: bytes, filename: str = "screenshot.jpg") -> str | None:
    """
    Upload an image to the OpenAI Files API and return its file_id for vision.
    """
    url = "https://api.openai.com/v1/files"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    files = {
        "file": (filename, image_bytes, "image/jpeg"),
        "purpose": (None, "vision"),
    }
    try:
        r = requests.post(url, headers=headers, files=files, timeout=60)
        if r.status_code != 200:
            # print and fall back; don't crash the run
            print(f"[Upload] error {r.status_code}: {r.text[:200]}")
            return None
        return r.json().get("id")
    except Exception as e:
        print(f"[Upload] exception: {e}")
        return None


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
    Responses API call with:
      - configurable timeouts (RESP_CONNECT_TIMEOUT / RESP_READ_TIMEOUT)
      - robust retries on server_error/overloaded/5xx and ReadTimeout
      - jittered exponential backoff
      - request-id printed on each retry
      - 'store': False to keep calls a bit lighter server-side
    """
    url = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    org = os.getenv("OPENAI_ORG")
    if org:
        headers["OpenAI-Organization"] = org

    # timeouts
    CONNECT_T = float(os.getenv("RESP_CONNECT_TIMEOUT", "20"))  # sec
    READ_T    = float(os.getenv("RESP_READ_TIMEOUT", "180"))    # sec (increase from 120)
    TIMEOUT   = (CONNECT_T, READ_T)

    # retries
    MAX_RETRIES = int(os.getenv("RESP_MAX_RETRIES", "5"))
    BASE = float(os.getenv("RESP_BACKOFF_BASE", "0.7"))

    # opt-in reasoning summaries and do not store
    payload = {
        **kwargs,
        "store": True,
        "reasoning": {
            "effort": os.getenv("REASONING_EFFORT", "medium"),
            "summary": os.getenv("REASONING_SUMMARY", "auto"),
        },
    }

    last_text = ""
    last_json = None
    last_req  = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _SESSION.post(url, headers=headers, json=payload, timeout=TIMEOUT)
            req_id = resp.headers.get("x-request-id")
            last_req = req_id

            # Try JSON
            try:
                data = resp.json()
                last_json = data
            except Exception:
                data = None
                last_text = (resp.text or "")[:800]

            # Success
            if resp.status_code == 200 and isinstance(data, dict) and "output" in data:
                return data

            # transient? server error family
            transient = (
                resp.status_code >= 500 or
                (isinstance(data, dict) and isinstance(data.get("error"), dict) and
                 data["error"].get("type") in {"server_error", "overloaded_error"})
            )

            if transient and attempt < MAX_RETRIES:
                delay = BASE * (1.7 ** (attempt - 1)) * random.uniform(0.85, 1.35)
                print(f"[Retry] {resp.status_code} server_error (req {req_id}) attempt {attempt}/{MAX_RETRIES}…")
                time.sleep(delay)
                continue

            if data is None:
                raise RuntimeError(f"API {resp.status_code} returned non-JSON response (req {req_id}):\n{last_text}")

            raise RuntimeError(f"API error {resp.status_code} (req {req_id}): {json.dumps(data)[:800]}")

        except requests.exceptions.ReadTimeout:
            # Treat ReadTimeout like transient; backoff & retry
            if attempt < MAX_RETRIES:
                delay = BASE * (1.9 ** (attempt - 1)) * random.uniform(0.9, 1.5)
                print(f"[Retry] read timeout (attempt {attempt}/{MAX_RETRIES})…")
                time.sleep(delay)
                continue
            raise RuntimeError(f"API ReadTimeout after {attempt} attempts (req {last_req})")
        except requests.exceptions.ConnectionError as e:
            # network hiccup — retry
            if attempt < MAX_RETRIES:
                delay = BASE * (1.9 ** (attempt - 1)) * random.uniform(0.9, 1.5)
                print(f"[Retry] connection error {e.__class__.__name__} (attempt {attempt}/{MAX_RETRIES})…")
                time.sleep(delay)
                continue
            raise RuntimeError(f"API connection error after {attempt} attempts (req {last_req}): {e}")

    # Fallback (shouldn't hit)
    raise RuntimeError(f"API error (req {last_req}): {last_text or json.dumps(last_json)[:800]}")


def check_blocklisted_url(url: str) -> None:
    """Raise ValueError if the given URL (including subdomains) is in the blocklist."""
    hostname = urlparse(url).hostname or ""
    if any(
        hostname == blocked or hostname.endswith(f".{blocked}")
        for blocked in BLOCKED_DOMAINS
    ):
        raise ValueError(f"Blocked URL: {url}")
