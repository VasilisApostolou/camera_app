#Helper functions for manipulating the frames
import cv2

def show_frame(window_name, frame):
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)  # allow resizing
    cv2.resizeWindow("Camera Feed", 1920, 1080) 
    cv2.imshow(window_name, frame)

def get_key_pressed(delay=1) -> int:
    return cv2.waitKey(delay) & 0xFF
