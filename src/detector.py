"""Detector: turns a webcam frame into facts. No event logic lives here."""
from __future__ import annotations

import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
MIN_VISIBILITY = 0.5


class Detector:
    """Runs MediaPipe face detection + pose estimation on each frame."""

    def __init__(self, face_conf: float = 0.6, pose_conf: float = 0.5) -> None:
        self.face = mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=face_conf
        )
        self.pose = mp_pose.Pose(
            min_detection_confidence=pose_conf, min_tracking_confidence=0.5
        )

    @staticmethod
    def _hands_raised(landmarks) -> bool:
        """True if either wrist is above its same-side shoulder.
        In image coordinates y grows downward, so smaller y = higher."""
        lm = landmarks.landmark
        for wrist, shoulder in ((LEFT_WRIST, LEFT_SHOULDER),
                                (RIGHT_WRIST, RIGHT_SHOULDER)):
            if (lm[wrist].visibility > MIN_VISIBILITY
                    and lm[shoulder].visibility > MIN_VISIBILITY
                    and lm[wrist].y < lm[shoulder].y):
                return True
        return False

    def process(self, frame):
        """Return (annotated_frame, facts_dict)."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = self.face.process(rgb)
        pose_res = self.pose.process(rgb)

        positions = []
        for det in face_res.detections or []:
            box = det.location_data.relative_bounding_box
            positions.append((round(box.xmin + box.width / 2, 3),
                              round(box.ymin + box.height / 2, 3)))
            x, y = int(box.xmin * w), int(box.ymin * h)
            cv2.rectangle(frame, (x, y),
                          (x + int(box.width * w), y + int(box.height * h)),
                          (0, 255, 0), 2)
        num_faces = len(positions)

        hands_raised = False
        if pose_res.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_res.pose_landmarks,
                                   mp_pose.POSE_CONNECTIONS)
            hands_raised = self._hands_raised(pose_res.pose_landmarks)

        # Pose tracks one person, so people = faces, or 1 if only a body is seen
        num_people = max(num_faces, 1 if pose_res.pose_landmarks else 0)

        facts = {
            "num_faces": num_faces,
            "num_people": num_people,
            "hands_raised": hands_raised,
            "people_positions": positions,
        }
        return frame, facts

    def close(self) -> None:
        self.face.close()
        self.pose.close()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = Detector()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            frame, facts = detector.process(frame)
            text = (f"faces={facts['num_faces']} people={facts['num_people']} "
                    f"hand_raised={facts['hands_raised']}")
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            cv2.imshow("VisionGuard - detector test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()