# main script : runs the camera feed
# python -m src.main

import cv2
import numpy as np
#helper functions and classes
from src.camera import open_camera, get_frame, release_camera
from src.utils import show_frame, get_key_pressed, get_screen_resolution
from src.processors import (apply_grayscale, apply_blur_filter,
                             detect_color, detect_shapes,
                             create_color_histogram)
from src.tracker import SORTTracker          
from src.yolo_detector import YoloDetector

def main():
    #open the cam with resolution 1600x900
    cap = open_camera(0, 1600, 900)
    
    current_mode = "normal" #default mode
    histogram_window_open = False
    first_frame = True
    
    # Start the YOLO detector and SORT tracker
    yolo = YoloDetector()
    tracker = SORTTracker(                   
        max_age=3,  #wait 3 frames before deleting a lost track                          
        min_hits=2, #Trust object ID until it has been seen for 2 frames                          
        iou_threshold=0.25  #Minimum IoU for matching detections to existing tracks                   
    )

    processing_modes = {
        "normal"    : lambda f: f,
        "grayscale" : apply_grayscale,
        "blur"      : apply_blur_filter,
        "yolo"      : lambda f: f,
    }

    #color boundaries
    color_ranges = {
        "Blue":  (np.array([100, 80,  50]), np.array([140, 255, 255])),
        "Red":   (np.array([0,   150, 100]), np.array([10,  255, 255])),
        "Green": (np.array([40,  80,  50]), np.array([80,  255, 255]))
    }

    #YOLO color boxes
    detection_color_map = {
        0:  (0,   255, 0),    # person      - green
        67: (255, 0,   0),    # cell phone  - blue
        2:  (0,   0,   255),  # car         - red
    }

    window_title = "Camera Feed"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    #find out how big is the camera feed
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #scale window to 95% of screen height and center
    screen_width, screen_height = get_screen_resolution()
    scale_factor  = (screen_height * 0.95) / cam_h
    window_height = int(cam_h * scale_factor)
    window_width  = int(cam_w * scale_factor)
    win_x = (screen_width  - window_width)  // 2
    win_y = (screen_height - window_height) // 2

    try:
        while True:
            ret, frame = get_frame(cap)
            if not ret:
                print("Failed to grab frame")
                break

            #Apply the selected filter (mode)
            display_frame = processing_modes.get(current_mode, lambda f: f)(frame)

            #COLOR MODE
            if current_mode == "color":
                detections = detect_color(frame, color_ranges)
                for name, bbox in detections:
                    bx, by, bw, bh = bbox
                    cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), (0,255,0), 2)
                    cv2.putText(display_frame, name, (bx, by-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            #SHAPES MODE
            if current_mode == "shapes":
                detect_shapes(display_frame)


            #YOLO & TRACKING MODE
            if current_mode == "yolo":
                #1. AI finds objects in the current frame.
                raw_detections = yolo.detect(frame)

                bbox_list = [d["bbox"] for d in raw_detections]

                #2. Update the tracker . It looks at the new boxes and tries
                #   to match them to existing tracks, and assigns IDs.
                tracked_objects = tracker.update(bbox_list)

                #3. Build a table matching track ID to its box coords.
                track_box_map = {}
                for track in tracker.tracks:
                    track_box_map[track.id] = track.get_state()

                def _iou(a, b):
                    # IoU (Intersection over Union) between two (x1,y1,x2,y2) boxes.
                    
                    #Find overlapping area's corners
                    xA, yA = max(a[0], b[0]), max(a[1], b[1])
                    xB, yB = min(a[2], b[2]), min(a[3], b[3])
                    
                    #Calculate overlap area
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    if inter == 0:
                        return 0.0 #no overlap
                    
                    #Calculate area of both boxes
                    areaA = (a[2]-a[0]) * (a[3]-a[1])
                    areaB = (b[2]-b[0]) * (b[3]-b[1])
                    
                    #Divide overlap by the total combined area
                    return inter / max(areaA + areaB - inter, 1e-6)


                #4. Match the AI's labels with the tracker's IDs
                for detection in raw_detections:
                    x1, y1, x2, y2 = detection["bbox"]
                    label    = detection["label"]
                    class_id = detection["class_id"]
                    box_color = detection_color_map.get(class_id, (0, 255, 255))

                    # Task : Find which tracker ID belongs to AI detection
                    # Solution : We find which trackerd box overlaps with this detection the most.
                    best_id  = None
                    best_iou = 0.0
                    for tid, tbox in track_box_map.items():
                        if tid not in tracked_objects:   
                            continue # track not yet confirmed = skip
                        
                        score = _iou(detection["bbox"], tbox)
                        if score > best_iou:
                            best_iou = score
                            best_id  = tid

                    # If we found match --> show ID.
                    obj_id = f"ID {best_id}" if best_id is not None else "Scanning..."

                    #Draw box and text.
                    label_with_id = f"{label} {obj_id}"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(display_frame, label_with_id, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # DISPLAY
            cv2.imshow(window_title, display_frame)

            if first_frame:
                cv2.resizeWindow(window_title, window_width, window_height)
                cv2.moveWindow(window_title, win_x, win_y)
                first_frame = False

            # HISTOGRAM MODE
            if current_mode == "histogram":
                hist_window_title = "Color Histogram"
                hist_canvas = create_color_histogram(frame)
                if not histogram_window_open:
                    cv2.namedWindow(hist_window_title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(hist_window_title, 1200, 800)
                    
                    hist_x = win_x + window_width + 20
                    cv2.moveWindow(hist_window_title, hist_x, win_y)
                    histogram_window_open = True
                
                cv2.imshow(hist_window_title, hist_canvas)
            
            #kill extra window if we switch mode
            elif histogram_window_open:
                cv2.destroyWindow("Color Histogram") 
                histogram_window_open = False

            # KEY HANDLING
            key_pressed = get_key_pressed()
            if key_pressed == ord('q'):
                break

            key_mode_mapping = {
                ord('n'): "normal",
                ord('g'): "grayscale",
                ord('b'): "blur",
                ord('c'): "color",
                ord('h'): "histogram",
                ord('s'): "shapes",
                ord('y'): "yolo"
            }

            if key_pressed in key_mode_mapping:
                current_mode = key_mode_mapping[key_pressed]
                print(f"Switched to {current_mode} mode")

    finally:
        #cleanup
        release_camera(cap)

if __name__ == "__main__":
    main()