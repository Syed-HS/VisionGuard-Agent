"""EventManager: turns per-frame facts into meaningful, de-duplicated events."""
from __future__ import annotations

from datetime import datetime


class EventManager:
    """Applies rules, debouncing and cooldowns. No camera or LLM code here."""

    def __init__(self, cooldown: float = 8.0, stable_frames: int = 5) -> None:
        self.cooldown = cooldown
        self.stable_frames = stable_frames
        self.last_fired: dict[str, float] = {}
        self.stable_people = 0      # last confirmed people count
        self._candidate = 0         # count we are currently "voting" on
        self._streak = 0            # consecutive frames showing the candidate

    def _ready(self, name: str, now: float) -> bool:
        """Cooldown gate: True (and records the time) if the event may fire."""
        last = self.last_fired.get(name)
        if last is not None and now - last < self.cooldown:
            return False
        self.last_fired[name] = now
        return True

    def _make(self, name: str, det: dict, now: float, description: str) -> dict:
        return {
            "event": name,
            "timestamp": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            "faces_detected": det["num_faces"],
            "people_detected": det["num_people"],
            "description": description,
        }

    def _people_change(self, count: int) -> tuple[int, int] | None:
        """Debounce: return (old, new) only once a new count is stable."""
        if count == self.stable_people:
            self._candidate, self._streak = count, 0
            return None
        if count == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = count, 1
        if self._streak >= self.stable_frames:
            old, self.stable_people = self.stable_people, count
            self._candidate, self._streak = count, 0
            return old, count
        return None

    def update(self, det: dict, now: float) -> list[dict]:
        """Consume one frame's facts, return any events that should fire."""
        events: list[dict] = []

        change = self._people_change(det["num_people"])
        if change:
            old, new = change
            name = "NEW_PERSON" if new > old else "PERSON_LEFT"
            if self._ready(name, now):
                events.append(self._make(
                    name, det, now, f"People count changed from {old} to {new}"))

        if det["num_faces"] > 1 and self._ready("MULTIPLE_FACES", now):
            events.append(self._make(
                "MULTIPLE_FACES", det, now,
                f"{det['num_faces']} faces detected in the frame"))

        if det["hands_raised"] and self._ready("HAND_RAISED", now):
            events.append(self._make(
                "HAND_RAISED", det, now, "A person raised a hand above shoulder level"))

        return events