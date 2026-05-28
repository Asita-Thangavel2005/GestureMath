class GestureRecognizer:

    def __init__(self):
        self.margin = 0.03

    def finger_up(self, landmarks, tip, pip):
        return landmarks.landmark[tip].y < landmarks.landmark[pip].y - self.margin

    def finger_down(self, landmarks, tip, pip):
        return landmarks.landmark[tip].y > landmarks.landmark[pip].y + self.margin

    def is_drawing_mode(self, landmarks):
        if not landmarks:
            return False
        index  = self.finger_up(landmarks, 8, 6)
        middle = self.finger_down(landmarks, 12, 10)
        ring   = self.finger_down(landmarks, 16, 14)
        pinky  = self.finger_down(landmarks, 20, 18)
        return index and middle and ring and pinky

    def is_fist(self, landmarks):
        if not landmarks:
            return False
        # FIXED: also require thumb down so thumbs-up doesn't trigger this
        return (
            self.finger_down(landmarks, 8, 6)
            and self.finger_down(landmarks, 12, 10)
            and self.finger_down(landmarks, 16, 14)
            and self.finger_down(landmarks, 20, 18)
            and self.finger_down(landmarks, 4, 3)   # thumb
        )

    def is_two_fingers_up(self, landmarks):
        if not landmarks:
            return False
        index  = self.finger_up(landmarks, 8, 6)
        middle = self.finger_up(landmarks, 12, 10)
        ring   = self.finger_down(landmarks, 16, 14)
        pinky  = self.finger_down(landmarks, 20, 18)
        # FIXED: also ensure index and middle are clearly separated (not just both up)
        index_tip  = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        spread = abs(index_tip.x - middle_tip.x)
        return index and middle and ring and pinky and spread > 0.02

    def is_thumbs_up(self, landmarks):
        if not landmarks:
            return False
        import math, numpy as np
        thumb_tip = np.array([landmarks.landmark[4].x, landmarks.landmark[4].y])
        thumb_mcp = np.array([landmarks.landmark[2].x, landmarks.landmark[2].y])
        angle = math.degrees(math.atan2(thumb_mcp[1] - thumb_tip[1],
                                        thumb_mcp[0] - thumb_tip[0]))
        if not (-110 < angle < -70):
            return False
        return (self.finger_down(landmarks, 8, 6) and
                self.finger_down(landmarks, 12, 10) and
                self.finger_down(landmarks, 16, 14) and
                self.finger_down(landmarks, 20, 18))