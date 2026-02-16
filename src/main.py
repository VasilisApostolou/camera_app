# main script : runs the camera feed
#python -m src.main

import cv2
import numpy as np
from src.camera import open_camera, get_frame, release_camera
from src.utils import show_frame, get_key_pressed
from src.processors import apply_grayscale, apply_blur_filter, detect_color
from src.tracker import ObjectTracker

def main():
    cap = open_camera(0, 1280, 720) #open webcam
    is_grayscale_active = False
    is_blur_active = False
    is_color_detection_active = False
    color_id = "blue"
    obj_tracker = ObjectTracker()
    color_ranges = {
        "Blue": (np.array([100, 80, 50]), np.array([140, 255, 255])),
        "Red": (np.array([0, 150, 100]), np.array([10, 255, 255])),
        "Green": (np.array([40, 80, 50]), np.array([80, 255, 255]))
    }
    
    try:
        while True:
            ret, frame = get_frame(cap)
            if not ret:
                print("Failed to grab frame")
                break
            
            #Only process if toggle is ON
            display_frame = frame.copy()
            if is_grayscale_active:
                display_frame = apply_grayscale(frame)
            
            elif is_blur_active:
                display_frame = apply_blur_filter(frame)
            
            elif is_color_detection_active:
                detections = detect_color(frame, color_ranges)
                for name, bbox in detections:
                    x, y, w, h = bbox
                    #draw rectangle
                    cv2.rectangle(display_frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                    #label each one
                    cv2.putText(display_frame, name, (x, y-50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            show_frame("Camera Feed", display_frame)
            

            #Handle Key Inputs
            key_pressed = get_key_pressed()
            if key_pressed == ord('q'): #QUIT
                break
            elif key_pressed == ord('g'): #APPLY GRAYSCALE
                is_grayscale_active = not is_grayscale_active
                print(f"Grayscale mode: {is_grayscale_active}")
            elif key_pressed == ord('b'): #APPLY BLUR
                is_blur_active = not is_blur_active
                print(f"Blur Filter: {is_blur_active}")
            elif key_pressed == ord('c'): #APPLY COLOR DETECTION
                is_color_detection_active = not is_color_detection_active
                print(f"Color Detection mode: {is_color_detection_active}")

            
    finally:
        print("Releasing camera ...")
        release_camera(cap) #always release resources


if __name__ == "__main__":
    main()