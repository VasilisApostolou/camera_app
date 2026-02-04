#All functions for handling the camera


#import libraries 
import cv2

def open_camera(camera_index=0, width=640, height=480): #camera_index=0 : default camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise Exception("Error: Could not open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    return ret, frame

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()