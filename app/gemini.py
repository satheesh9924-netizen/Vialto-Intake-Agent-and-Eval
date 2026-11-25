"""Real model: Google Gemini through its plain REST API (httpx; no extra SDK to install).

Set GEMINI_API_KEY (free key from https://aistudio.google.com). Optional environment variables:

  GEMINI_MODEL       model name (default: gemini-flash-lite-latest; pick one your key lists in AI Studio)
  GEMINI_RPM         max real requests per minute (default 10: keeps a free key under its limit)
  GEMINI_CACHE_DIR   where replies are cached (default <repo>/.cache/gemini; GEMINI_CACHE=off disables it)

Three things make a free key workable for an evaluation suite:

1. THROTTLE   one shared limiter spaces real requests (free tiers allow only a few per minute).
2. RETRY      429 (rate limited) and 5xx (server) answers are retried with backoff. A *daily* quota
              that is used up stops the run with a clear message instead of silently flagging every
              document, which would produce misleading metrics.
3. CACHE      a reply is stored on disk, so re-running the same evaluation costs no quota and gives
              the same numbers. The cache key includes "the Nth time this client asked this exact
              question", so the consistency test (same input, 5 runs) still gets 5 genuine samples
              and not one answer copied five times.
"""
import hashlib
import json
import os
import pathlib
import threading
import time

import httpx

from .llm import LLMTimeout

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-flash-lite-latest"


class GeminiConfigError(RuntimeError):
    """Missing/invalid key, unknown model, or other problem only the user can fix. Stops the run."""


class GeminiQuotaExhausted(RuntimeError):
    """The daily free quota is used up. Stops the run; try again after midnight Pacific time or use a cache."""


class _Throttle:
    """Process-wide limiter: at most `rpm` real requests per minute."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self, rpm: float):
        gap = 60.0 / max(rpm, 0.1)
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + gap
        if start > now:
            time.sleep(start - now)


_THROTTLE = _Throttle()


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying: the server's own hint if given, else exponential backoff."""
    hint = resp.headers.get("retry-after")
    if hint and hint.replace(".", "", 1).isdigit():
        return min(float(hint), 90)
    try:
        for d in resp.json().get("error", {}).get("details", []):
            delay = d.get("retryDelay")
            if delay and delay.endswith("s"):
                return min(float(delay[:-1]), 90)
    except Exception:
        pass
    return min(2 ** attempt * 3, 60)


class GeminiClient:
    def __init__(self, model: str | None = None, api_key: str | None = None, http: httpx.Client | None = None,
                 rpm: float | None = None, cache_dir: str | None = None, max_retries: int = 5,
                 sleep=time.sleep, throttle: bool = True):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise GeminiConfigError("GEMINI_API_KEY is not set. Create a free key at https://aistudio.google.com "
                                    "and run: export GEMINI_API_KEY=...")
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.http = http or httpx.Client(timeout=60)
        self.rpm = rpm or float(os.getenv("GEMINI_RPM", "10"))
        self.max_retries, self.sleep, self.throttle = max_retries, sleep, throttle
        off = os.getenv("GEMINI_CACHE", "on").lower() == "off"
        self.cache_dir = None if off else pathlib.Path(cache_dir or os.getenv("GEMINI_CACHE_DIR", ROOT / ".cache" / "gemini"))
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._seen: dict[str, int] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0}

    # ------------------------------------------------------------------ cache
    def _key(self, system: str, user: str) -> str:
        base = hashlib.sha256(json.dumps([self.model, system, user]).encode()).hexdigest()
        nth = self._seen.get(base, 0)           # "the Nth time THIS client asked this exact question"
        self._seen[base] = nth + 1
        return f"{base}-{nth}"

    # ------------------------------------------------------------------ one request
    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        key = self._key(system, user)
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            self.stats["cache_hits"] += 1
            return json.loads(path.read_text())["text"]
        text = self._request(system, user, max_tokens)
        if path:
            path.write_text(json.dumps({"model": self.model, "text": text}))
        return text

    def _request(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]} if system else None,
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"},
        }
        body = {k: v for k, v in body.items() if v is not None}
        url = ENDPOINT.format(model=self.model)
        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            if self.throttle:
                _THROTTLE.wait(self.rpm)
            self.stats["calls"] += 1
            try:
                resp = self.http.post(url, json=body, headers={"x-goog-api-key": self.api_key})
            except httpx.TransportError as e:
                last = f"network error: {e}"
                self.stats["retries"] += 1
                self.sleep(min(2 ** attempt * 2, 30))
                continue
            if resp.status_code == 200:
                return self._text(resp.json())
            detail = self._error_text(resp)
            if resp.status_code in (400, 401, 403, 404):
                raise GeminiConfigError(f"Gemini rejected the request (HTTP {resp.status_code}): {detail[:400]}. "
                                        f"Check GEMINI_API_KEY and GEMINI_MODEL (current: {self.model}).")
            if resp.status_code == 429:
                if "perday" in detail.replace("_", "").replace(" ", "").lower():
                    raise GeminiQuotaExhausted(f"Daily free quota used up for {self.model}: {detail[:400]}")
                last = f"rate limited: {detail[:200]}"
            elif resp.status_code >= 500:
                last = f"server error {resp.status_code}: {detail[:200]}"
            else:
                raise GeminiConfigError(f"Unexpected HTTP {resp.status_code}: {detail[:400]}")
            self.stats["retries"] += 1
            self.sleep(_retry_delay(resp, attempt))
        raise LLMTimeout(f"Gemini unavailable after {self.max_retries + 1} attempts ({last})")

    @staticmethod
    def _error_text(resp: httpx.Response) -> str:
        try:
            return json.dumps(resp.json().get("error", resp.json()))
        except Exception:
            return resp.text

    @staticmethod
    def _text(data: dict) -> str:
        """Join the text parts. A blocked or empty answer becomes "" (the agent treats it as a bad reply)."""
        cands = data.get("candidates") or []
        if not cands:                      # e.g. promptFeedback.blockReason
            return ""
        parts = (cands[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))


class GeminiLLM:
    """Same `generate(system, user) -> str` shape as every other model, so the agent does not change."""

    def __init__(self, model: str | None = None, **kw):
        self.client = GeminiClient(model, **kw)
        self.profile = f"gemini:{self.client.model}"

    def generate(self, system: str, user: str) -> str:
        return self.client.generate(system, user)
