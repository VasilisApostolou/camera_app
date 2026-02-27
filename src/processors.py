import cv2 
import numpy as np 
from typing import Optional, Tuple, List, Dict
from matplotlib import pyplot as plt


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
        kern = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kern, iterations=2) #shrinks white regions , removes dots
        mask = cv2.dilate(mask, kern, iterations=2) # expands white regions, restores main object size

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

def create_color_histogram(frame: np.ndarray) -> np.ndarray:
    #create a black canvas to draw the graph
    hist_width, hist_height = 512,400
    canvas = np.zeros((hist_height,hist_width, 3), dtype=np.uint8) #creates black image

    #define the colors we need to analyze
    colors = ("b", "g", "r")

    for i,col in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0,256])
        ''' [frame] : image input
            [i] : which color channel
            [None] : no mask (use the full image)
            [256] : number of bins
            [0-256] : intensity range
        '''

        cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX) #minimize values to fit histogram
        
        hist = hist.flatten() #convert to 1D array for easier plotting
        #draw the histogram lines
        for x in range(1,256):
            pt1 = ( (x-1)*2, hist_height - int(hist[x-1]) )
            pt2 = ( x*2, hist_height - int(hist[x]) )

            line_color = (255 if col == "b" else 0,
                        255 if col == "g" else 0,
                        255 if col == "r" else 0)
            cv2.line(canvas, pt1, pt2, line_color, 2)
    return canvas

def detect_shapes(frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    # Adjusted Canny thresholds for better edge detection
    edged = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_results = []
    #process each contour to determine shape
    for contour in contours:
        area = cv2.contourArea(contour)
        print(f"Contour area: {area}")
        if area < 5000:  
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.05:
            continue
        #approximate contour shape by simplifying it to a polygon
        # Increased epsilon for better approximation
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

        #draw the contour
        cv2.drawContours(frame, [contour], 0, (0,255,0), 2)

        #find center of the shape for labeling
        M = cv2.moments(contour) #computes spatial moments of the contour
        if M["m00"] == 0: #skip if we can't calculate center
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        #detect shape type based on number of vertices
        sides = len(approx)
        if sides == 3:
            shape_type = "Triangle"
        elif sides == 4:
            shape_type = "Rectangle"
        elif sides == 5:
            shape_type = "Pentagon"
        elif sides == 6:
            shape_type = "Hexagon"
        
        else:
            peri = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (peri * peri) 
            if circularity > 0.8:
                shape_type = "Circle"
            else:
                shape_type = "."
        cv2.putText(frame, shape_type, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        shape_results.append((shape_type, approx))
    return shape_results