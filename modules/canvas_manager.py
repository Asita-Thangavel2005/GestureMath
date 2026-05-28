import numpy as np
import cv2
from collections import deque
import config  # FIXED: import config so color/thickness stay in sync

class CanvasManager:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.color = config.DRAW_COLOR       # FIXED: use config color
        self.thickness = config.THICKNESS    # FIXED: use config thickness
        self.reset()

    def reset(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.strokes = []
        self.current_stroke = deque(maxlen=1024)  # increased from 512

    def start_new_stroke(self):
        if len(self.current_stroke) > 1:
            self.strokes.append(list(self.current_stroke))
        self.current_stroke.clear()
        self.redraw_all()

    def add_point(self, x, y):
        # FIXED: interpolate between last point and new point for smoother lines
        if self.current_stroke:
            last_x, last_y = self.current_stroke[-1]
            dist = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
            if dist > 1:
                steps = max(1, int(dist / 3))
                for i in range(1, steps + 1):
                    ix = int(last_x + (x - last_x) * i / steps)
                    iy = int(last_y + (y - last_y) * i / steps)
                    self.current_stroke.append((ix, iy))
                return
        self.current_stroke.append((int(x), int(y)))

    def erase_last_stroke(self):
        # FIXED: always save current stroke first before erasing
        if len(self.current_stroke) > 1:
            self.strokes.append(list(self.current_stroke))
        self.current_stroke.clear()

        if self.strokes:
            self.strokes.pop()

        self.redraw_all()

    def redraw_all(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for stroke in self.strokes:
            if len(stroke) > 1:
                points = np.array(stroke, dtype=np.int32)
                cv2.polylines(self.canvas, [points], False,
                              self.color, self.thickness,
                              lineType=cv2.LINE_AA)  # FIXED: anti-aliased lines

    def get_canvas(self):
        temp = self.canvas.copy()
        if len(self.current_stroke) > 1:
            points = np.array(self.current_stroke, dtype=np.int32)
            cv2.polylines(temp, [points], False,
                          self.color, self.thickness,
                          lineType=cv2.LINE_AA)  # FIXED: anti-aliased
        return temp

    def clear(self):
        self.reset()