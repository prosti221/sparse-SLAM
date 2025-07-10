import numpy as np


class Point:
    def __init__(self, pt_3d, color=None, descriptor=None):
        """
        A point in 3D space. It will have its own 3D coordinates, and a color.
        """
        self.pt_3d = pt_3d
        self.color = color
        self.descriptor = descriptor
