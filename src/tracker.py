import numpy as np 
from typing import Optional, Tuple

class ObjectTracker:
    def __init__(self):
        self.prev_center = None
    
    def update(self, bbox):
        #bbox = (x, y, w, h)
        x,y,w,h = bbox

        #compute center
        cx = x + w // 2
        cy = y + h // 2

        #compute velocity
        if self.prev_center is None:
            dx, dy = 0,0
        else:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]

        self.prev_center = (cx, cy)

        return (cx, cy), (dx, dy)