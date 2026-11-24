# VisionGuard Agent

Real-time webcam monitoring that turns computer-vision detections into
actionable alerts using a local LLM (Ollama). No cloud, no API keys.

![demo](demos/screenshot.png)

## What it does
- Detects faces and body pose with MediaPipe + OpenCV
- Converts raw detections into events: `NEW_PERSON`, `PERSON_LEFT`,
  `MULTIPLE_FACES`, `HAND_RAISED`
- Sends each event to a local LLM (llama3.2:3b via Ollama) and gets back a
  validated JSON alert with severity, recommended action and confidence
- Logs every event, LLM response and latency to CSV

## Architecture
```
Webcam -> detector.py (facts) -> event_manager.py (rules, debounce, cooldown)
       -> llm_agent.py (background thread, Ollama, JSON validation)
       -> logger.py (console + CSV) -> video overlay
```

## Design decisions
- **LLM only on events, not every frame:** keeps the system fast and cheap.
- **Background worker thread:** the video never freezes while the LLM thinks.
- **Debounce + cooldown:** people-count changes must be stable for 5 frames;
  each event type has an 8 s cooldown, preventing alert spam.
- **Validated LLM output:** JSON mode, severity/confidence checks, and a
  fallback response if Ollama is down or returns bad data.
- **Prompt iteration:** added explicit severity rules after the model
  over-escalated hand raises.
- **Latency logging** for observability.

## Setup
Requires Python 3.10 to 3.12 and [Ollama](https://ollama.com).
```bash
git clone https://github.com/Syed-HS/VisionGuard-Agent.git
cd VisionGuard-Agent
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
python -m src.main           # press q to quit
python -m pytest             # run tests
```

## Example log (logs/events.csv)
```
timestamp,event,faces,people,alert,severity,action,confidence,latency_ms
2026-11-24 18:46:36,NEW_PERSON,1,1,New person detected,low,Monitor the person's behavior and ensure their safety,0.9,622
2026-11-24 18:46:37,HAND_RAISED,1,1,"Person raised hand, may need attention",low,Check on the person and offer assistance if needed,0.8,649
2026-11-24 18:46:46,HAND_RAISED,1,1,"A person raised a hand above shoulder level, may need help or attention",low,Check on the person and offer assistance if needed,0.9,987
```

## Limitations
- MediaPipe Pose tracks one person, so people count is derived from faces.
- A 3B model is fast but sometimes imprecise; severity rules in the prompt
  keep it consistent.
- Single camera, no identity recognition.

## Future work
- Multi-person pose with the MediaPipe Tasks API
- Multiple cameras and a web dashboard
- Event snapshots saved alongside the log
