# probe_reasoning.py
import os, json, requests
from dotenv import load_dotenv

load_dotenv(override=True)

URL = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json",
}
ORG = os.getenv("OPENAI_ORG")
if ORG:
    HEADERS["OpenAI-Organization"] = ORG

payload = {
    "model": os.getenv("MODEL", "computer-use-preview"),
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text",
                         "text": "Plan a 3-step approach to compare two news articles for bias and write a short rationale for each step."}]
        }
    ],
    "truncation": "auto",
    "reasoning": {
        "effort": os.getenv("REASONING_EFFORT", "medium"),
        "summary": os.getenv("REASONING_SUMMARY", "auto"),
    },
}

resp = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
try:
    data = resp.json()
except Exception:
    print("status:", resp.status_code)
    print((resp.text or "")[:800])
    raise

print("status:", resp.status_code)
print(json.dumps(data, indent=2)[:4000])
