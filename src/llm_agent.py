"""LLMAgent: sends events to a local Ollama model and validates the JSON reply."""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import ollama

MODEL = "llama3.2:3b"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"
DEFAULT_PROMPT = (
    "You are a security and monitoring assistant. Respond ONLY with JSON: "
    '{"alert": str, "severity": "low"|"medium"|"high", "action": str, '
    '"confidence": float 0-1}. Be concise.'
)
VALID_SEVERITY = {"low", "medium", "high"}


def validate(raw: str) -> dict:
    """Parse and sanitize the model output. Raises on unusable input."""
    data = json.loads(raw)
    severity = str(data.get("severity", "low")).lower()
    if severity not in VALID_SEVERITY:
        severity = "low"
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "alert": str(data["alert"]),
        "severity": severity,
        "action": str(data.get("action", "Review the event")),
        "confidence": round(confidence, 2),
    }


class LLMAgent:
    """Non-blocking agent: submit() events, poll_results() finished ones."""

    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        text = PROMPT_PATH.read_text(encoding="utf-8").strip() if PROMPT_PATH.exists() else ""
        self.system_prompt = text or DEFAULT_PROMPT
        self._jobs: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def analyze(self, event: dict) -> dict:
        """Blocking call to Ollama. Always returns a valid result dict."""
        start = time.perf_counter()
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(event)},
                ],
                format="json",
                options={"temperature": 0.2},
            )
            result = validate(resp["message"]["content"])
        except Exception as exc:  # network, parse, schema: all fall back
            result = {
                "alert": "LLM unavailable",
                "severity": "low",
                "action": f"Check Ollama ({type(exc).__name__})",
                "confidence": 0.0,
            }
        result["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return result

    def _worker(self) -> None:
        try:  # warm-up so the first real event is not slow
            ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}])
        except Exception:
            pass
        while True:
            event = self._jobs.get()
            if event is None:
                break
            self._results.put((event, self.analyze(event)))

    def submit(self, event: dict) -> None:
        self._jobs.put(event)

    def poll_results(self) -> list[tuple[dict, dict]]:
        """Return finished (event, result) pairs without blocking."""
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        self._jobs.put(None)


if __name__ == "__main__":
    agent = LLMAgent()
    agent.submit({"event": "MULTIPLE_FACES", "faces_detected": 2,
                  "timestamp": "2026-11-24 18:30:00",
                  "description": "Two faces detected in the frame"})
    while not (res := agent.poll_results()):
        time.sleep(0.2)
    print(res[0][1])
    agent.stop()