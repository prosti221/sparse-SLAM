import numpy as np


class Map:
    def __init__(self):
        self.points = []       # All known 3D points
        self.keyframes = []    # All added keyframes

    def add_points(self, points):
        self.points += points

    def add_keyframe(self, kf):
        self.keyframes.append(kf)
