"""Main loop: webcam -> detector -> events -> LLM (background) -> log + overlay."""
from __future__ import annotations

import time

import cv2

from src.detector import Detector
from src.event_manager import EventManager
from src.llm_agent import LLMAgent
from src.logger import log_event

COLORS = {"low": (0, 200, 0), "medium": (0, 165, 255), "high": (0, 0, 255)}  # BGR


def main() -> None:
    cap = cv2.VideoCapture(0)
    detector, manager, agent = Detector(), EventManager(), LLMAgent()
    banner, banner_color = "Monitoring...", (255, 255, 255)
    prev = time.time()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame, facts = detector.process(frame)

            now = time.time()
            for event in manager.update(facts, now):
                agent.submit(event)
                banner, banner_color = f"{event['event']}: analyzing...", (255, 255, 0)

            for event, result in agent.poll_results():
                log_event(event, result)
                banner = f"[{result['severity'].upper()}] {result['alert']}"
                banner_color = COLORS[result["severity"]]

            fps = 1 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, banner[:70], (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2)
            cv2.imshow("VisionGuard Agent", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        agent.stop()
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()