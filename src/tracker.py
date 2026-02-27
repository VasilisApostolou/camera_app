import numpy as np 
from typing import Optional, Tuple

class ObjectTracker:
    def __init__(self):
        self.objects = {} #store IDs as keys
        self.next_id = 0 #counter for assigning new IDs
    
    def update(self, detections):
        #bbox = (x, y, w, h)
        new_objects = {}
        
        for bbox in detections:
            x,y,w,h = bbox
            cx = x + w // 2
            cy = y + h // 2
            #centroid tracking -> find closest existing object to this centroid
            closest_id = None
            min_distance = 50 #max pixels an object can move between frames
            for obj_id,center in self.objects.items():
                calculated_distance = np.hypot(cx - center[0], cy - center[1]) #euclidean distance
                if calculated_distance < min_distance:
                    min_distance = calculated_distance
                    closest_id = obj_id
            if closest_id is not None:
                new_objects[closest_id] = (cx, cy) #update existing object
                del self.objects[closest_id] #remove from old list
            else:
                new_objects[self.next_id] = (cx, cy) #add new object
                self.next_id += 1
        self.objects = new_objects #replace old objects with updated ones
        return self.objects