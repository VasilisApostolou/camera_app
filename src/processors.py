import cv2 
import numpy as np 
from typing import Optional, Tuple

def apply_grayscale(frame: np.ndarray) -> np.darray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_blur_filter(frame: np.darray, kernel_size: int = 15) -> np.darray:
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def detect_color(frame: np.darray) -> Optional[Tuple[int, int, int, int]]: #return tuple of int or NONE
    #convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    #define the range of the desired color
    lower_bound = np.array([100,150,50])
    upper_bound = np.array([140,255,255])

    #create a mask : pixels in range -> white else -> black
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    #find outlines
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #RETR_EXTERNAL means we look for outer border we dont care about holes inside the object
    #CHAIN_APPROX_SIMPLE means if points are in a straight line keep only endpoints -> less data same shape

    if outlines:
        largest_outline = max(outlines, key=cv2.contourArea) #avoid small dots and noise
        if cv2.contourArea(largest_outline) > 500: 
            return cv2.boundingRect(largest_outline)

    return None