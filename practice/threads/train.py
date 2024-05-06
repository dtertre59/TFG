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

class ProcessFrameThread(threading.Thread):
    def __init__(self, camera_thread):
        threading.Thread.__init__(self)
        self.camera_thread = camera_thread
        self.running = False
    
    def process_frame(self):
        self.running = True
        while self.running:
            if self.camera_thread.send_frame:
                with self.camera_thread.lock:
                    frame = self.camera_thread.frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Processed Frame', gray_frame)
                self.camera_thread.send_frame = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    def run(self):
        self.process_frame()

def trigger_func():
    time.sleep(5)  # Espera 5 segundos
    return True  # Devuelve True después de 5 segundos

def run_trigger():
    trigger_thread = threading.Thread(target=trigger_func)
    trigger_thread.start()
    trigger_thread.join()

if __name__ == "__main__":
    cam_thread = CameraThread(0, trigger_func=run_trigger)
    cam_thread.start()

    process_thread = ProcessFrameThread(cam_thread)
    process_thread.start()

    cam_thread.join()
