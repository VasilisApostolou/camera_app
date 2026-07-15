#All functions for handling the camera

#import libraries 
import cv2
from threading import Thread, Lock

class ThreadedCamera:
    def __init__(self, camera_index=0, width=1600,height=900):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.ret, self.frame = self.cap.read() #grab first frame

        #thread control flags
        self.started = False
        self.read_lock = Lock() #Prevents main loop and thread from modifying the frame at the same time

    def start(self):
        if self.started:
            return self
        self.started = True
        #create background thread that runs the "update" method
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True #thread closes automatically when main program closes
        self.thread.start()
        return self
    
    def update(self):
        #keep grabbing frames while looping in the background
        while self.started:
            ret,frame = self.cap.read()
            with self.read_lock:
                self.ret = ret
                if ret:
                    self.frame = frame
    
    def get_frame(self):
        #read the frame to pass to main.py
        with self.read_lock:
            return self.ret, self.frame.copy()
    
    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

    