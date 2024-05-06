import cv2
import threading
import numpy as np
import time

class CameraThread(threading.Thread):
    def __init__(self, cam_id, trigger_func=None):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.send_frame = False
        self.trigger_func = trigger_func
    
    def capture_frames(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
                cv2.imshow('Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif self.trigger_func is not None and self.trigger_func():
                    self.send_frame = True
        self.cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        self.capture_frames()
    
    def stop_capture(self):
        self.running = False


def run_trigger():
    print(1)

if __name__ == "__main__":
    cam_thread = CameraThread(0, trigger_func=run_trigger)
    cam_thread.start()

    
    print('aaaaaaaaaaaaaaaaaaaaaa')
    cam_thread.join()
    print('bbbbbbbbbbbbbb')