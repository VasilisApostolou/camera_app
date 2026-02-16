#Helper functions for manipulating the frames
import cv2
import subprocess


def show_frame(window_name, frame):
    cv2.imshow(window_name, frame)

def get_key_pressed(delay=1) -> int:
    return cv2.waitKey(delay) & 0xFF

def get_screen_resolution():
    output = subprocess.check_output("xdpyinfo | grep dimensions", shell=True).decode()
    dims = output.split()[1]  # e.g. "1920x1080"
    width, height = dims.split("x")
    return int(width), int(height)
