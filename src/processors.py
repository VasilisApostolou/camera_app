import cv2 
import numpy as np 
from typing import Optional, Tuple, List, Dict

def apply_grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_blur_filter(frame: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def detect_color(
    frame: np.ndarray, 
    color_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[str, Tuple[int, int, int, int]]]:   
    #convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []
    for color_name, (lower,upper) in color_ranges.items():

        #create a mask : pixels in range -> white else -> black
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        #clean the noise of the frame using morphological opening
        mask = cv2.erode(mask, None, iterations=2) #shrinks white regions , removes dots
        mask = cv2.dilate(mask, None, iterations=2) # expands white regions, restores main object size

        #find outlines
        outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #RETR_EXTERNAL means we look for outer border we dont care about holes inside the object
        #CHAIN_APPROX_SIMPLE means if points are in a straight line keep only endpoints -> less data same shape

        if outlines:
            largest_outline = max(outlines, key=cv2.contourArea) #avoid small dots and noise
            if cv2.contourArea(largest_outline) > 500: 
                bbox = cv2.boundingRect(largest_outline) #convert blob to rectangle
                detections.append((color_name, bbox))

    return detections