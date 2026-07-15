import time
import subprocess
import os
import requests

class AlertSystem:
    def __init__(self,cooldown=5.0, sound_file="alert.wav"):
        self.cooldown = cooldown
        self.sound_file = sound_file
        self.last_alert_time = 0.0
        self.topic = "vasilis_alert_99"

        #start audio path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.sound_path = os.path.join(current_dir, sound_file)
    
    def process_detections(self,detections):
        person_detected = any(d["label"] == "person" for d in detections)

        if person_detected:
            current_time = time.time()

            if (current_time - self.last_alert_time) > self.cooldown:
                self._trigger_audio()
                self._send_phone_notification()
                self.last_alert_time = current_time

    def _trigger_audio(self):
        print("Person Detected!")
        if os.path.exists(self.sound_path):
            subprocess.Popen(["paplay", self.sound_file])
        else:
            print(f"no audio file named {self.sound_file} found")
    
    def _send_phone_notification(self):
        print("Pinging phone...")
        try:
            #send a simple POST request to the ntfy server
            requests.post(f"https://ntfy.sh/{self.topic}",
                data="A person was detected by the camera!".encode('utf-8'),
                headers={
                    "Title": "Camera Alert",
                    "Priority": "high",   
                    "Tags": "warning,eyes"    
                }
            )
        except Exception as e:
            print(f"Failed to reach ntfy: {e}")