"""Logger: console output plus an append-only CSV audit trail."""
from __future__ import annotations

import csv
from pathlib import Path

LOG_PATH = Path("logs") / "events.csv"
FIELDS = ["timestamp", "event", "faces", "people", "alert",
          "severity", "action", "confidence", "latency_ms"]


def log_event(event: dict, result: dict, path: Path = LOG_PATH) -> None:
    row = {
        "timestamp": event["timestamp"],
        "event": event["event"],
        "faces": event["faces_detected"],
        "people": event["people_detected"],
        "alert": result["alert"],
        "severity": result["severity"],
        "action": result["action"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
    }
    print(f"[{row['timestamp']}] {row['event']:<15} "
          f"{row['severity'].upper():<6} {row['alert']} -> {row['action']} "
          f"({row['latency_ms']} ms)")
    path.parent.mkdir(exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)