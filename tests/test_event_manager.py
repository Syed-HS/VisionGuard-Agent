from src.event_manager import EventManager


def det(people=0, faces=None, hand=False):
    faces = people if faces is None else faces
    return {"num_faces": faces, "num_people": people,
            "hands_raised": hand, "people_positions": []}


def names(events):
    return [e["event"] for e in events]


def test_new_person_needs_stable_frames():
    m = EventManager(stable_frames=5)
    for i in range(4):
        assert m.update(det(1), 100 + i) == []
    assert names(m.update(det(1), 104)) == ["NEW_PERSON"]


def test_flicker_is_ignored():
    m = EventManager(stable_frames=5)
    out = []
    for i in range(20):
        out += m.update(det(1 if i % 2 == 0 else 0), 100 + i * 0.03)
    assert out == []


def test_person_left():
    m = EventManager(stable_frames=5)
    for i in range(5):
        m.update(det(1), 100 + i)
    out = []
    for i in range(5):
        out += m.update(det(0), 200 + i)
    assert names(out) == ["PERSON_LEFT"]


def test_cooldown_blocks_repeats():
    m = EventManager(cooldown=8)
    assert names(m.update(det(1, hand=True), 100.0)) == ["HAND_RAISED"]
    assert m.update(det(1, hand=True), 101.0) == []
    assert names(m.update(det(1, hand=True), 109.0)) == ["HAND_RAISED"]


def test_multiple_faces():
    m = EventManager()
    assert "MULTIPLE_FACES" in names(m.update(det(2), 100.0))


def test_event_schema():
    m = EventManager()
    event = m.update(det(2), 100.0)[0]
    assert set(event) == {"event", "timestamp", "faces_detected",
                          "people_detected", "description"}