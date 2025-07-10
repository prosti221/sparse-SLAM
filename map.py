import numpy as np


class Map:
    def __init__(self):
        self.points = []       # All known 3D points
        self.point_coords = set()  # Track coordinates to avoid duplicates
        self.keyframes = []    # All added keyframes

    def add_points(self, points):
        new_points = []
        for pt in points:
            # Create a hashable key from coordinates (rounded to avoid floating point issues)
            coord_key = tuple(np.round(pt.pt_3d, 3))
            if coord_key not in self.point_coords:
                self.point_coords.add(coord_key)
                new_points.append(pt)

        self.points.extend(new_points)  # Use extend instead of +=
        # print(f"Added {len(new_points)} new points, total: {len(self.points)}")

    def add_keyframe(self, kf):
        self.keyframes.append(kf)
