# main script : runs the camera feed
# python -m src.main

import cv2
import numpy as np
from src.camera import open_camera, get_frame, release_camera
from src.utils import show_frame, get_key_pressed, get_screen_resolution
from src.processors import apply_grayscale, apply_blur_filter, detect_color, create_color_histogram
from src.tracker import ObjectTracker

def main():
    # Set camera resolution
    cap = open_camera(0, 1600, 900) 
    current_mode = "normal"
    histogram_window_open = False
    first_frame = True
    
    processing_modes = {
        "normal" : lambda f: f,
        "grayscale": apply_grayscale,
        "blur": apply_blur_filter,
    }
    
    color_ranges = {
        "Blue": (np.array([100, 80, 50]), np.array([140, 255, 255])),
        "Red": (np.array([0, 150, 100]), np.array([10, 255, 255])),
        "Green": (np.array([40, 80, 50]), np.array([80, 255, 255]))
    }

    # 1. WINDOW SETUP (Crucial for "Bigger" window)
    window_title = "Camera Feed"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    screen_width, screen_height = get_screen_resolution()

    # Calculate scaling to 85% of screen height
    scale_factor = (screen_height * 0.95) / cam_h
    window_height = int(cam_h * scale_factor)
    window_width = int(cam_w * scale_factor)

    win_x = (screen_width - window_width) // 2
    win_y = (screen_height - window_height) // 2
    
    try:
        while True:
            hist_window_title = "Color Histogram"
            cv2.namedWindow(hist_window_title, cv2.WINDOW_NORMAL) # Ensure histogram window is resizable
            ret, frame = get_frame(cap)
            if not ret:
                print("Failed to grab frame")
                break   
            
            display_frame = processing_modes.get(current_mode, lambda f: f)(frame)
            
            if current_mode == "color":
                detections = detect_color(frame, color_ranges)
                for name, bbox in detections:
                    # These were shadowing your window coordinates
                    bx, by, bw, bh = bbox
                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(display_frame, name, (bx, by - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(window_title, display_frame)
            
            # Initial resize and move
            if first_frame:
                cv2.resizeWindow(window_title, window_width, window_height)
                cv2.moveWindow(window_title, win_x, win_y)
                first_frame = False

            # Histogram Handling
            if current_mode == "histogram":
                hist_canvas = create_color_histogram(frame)
                cv2.imshow(hist_window_title, hist_canvas)
                if not histogram_window_open:
                    cv2.resizeWindow(hist_window_title, 1200, 800)
                    # Position it to the right of the camera feed
                    hist_x = win_x + window_width + 20
                    hist_y = win_y
                    cv2.moveWindow(hist_window_title, hist_x, hist_y)
                    histogram_window_open = True
            elif histogram_window_open:
                cv2.destroyWindow("Color Histogram")
                histogram_window_open = False   

            # Key Handling
            key_pressed = get_key_pressed()
            if key_pressed == ord('q'): 
                break

            key_mode_mapping = {
                ord('n'): "normal",
                ord('g'): "grayscale",
                ord('b'): "blur",
                ord('c'): "color",
                ord('h'): "histogram"
            }
            
            if key_pressed in key_mode_mapping:
                current_mode = key_mode_mapping[key_pressed]
                print(f"Switched to {current_mode} mode")
            
    finally:
        release_camera(cap)

if __name__ == "__main__":
    main()