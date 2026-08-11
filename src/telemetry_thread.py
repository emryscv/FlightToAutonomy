import cv2 
import os
import subprocess
import math
import numpy as np

from src.GateDetector import GateDetector
import utils_io

gate_detector = GateDetector()

def telemetry_thread(tello):
    frame_read = tello.get_frame_read()

    while True:
        telemetry = tello.get_current_state()
        telemetry["speed"] = math.sqrt(
            telemetry["vgx"] * telemetry["vgx"] + 
            telemetry["vgy"] * telemetry["vgy"] + 
            telemetry["vgz"] * telemetry["vgz"]
        )

        subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)
        print(f"Battery {telemetry["bat"]} %")
        print(f"Altitude {telemetry["h"]} cm")
        print(f"Speed {telemetry["speed"]} m/s")
        print(f"Temperature {(telemetry["templ"] + telemetry["temph"]) / 2} °C")
        print(f"Flight time {telemetry["time"]} s")

        img = frame_read.frame
        #Tello uses BGR format, but OpenCV uses RGB format, so we need to convert it
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Predict the gate position using the GateDetector
        prediction = gate_detector.predict(img)
        
        # pred_img, bboxes = utils_io.display_target(np.float32(prediction[0]), 
        #                                            img, 
        #                                            gate_detector.config['output_shape'],
        #                                            ret=True)
        
        #Display the gate position on the image
        pred_img, bboxes = GateDetector.display_predicticed_gate_center(np.float32(prediction[0]), 
                                                         img, 
                                                         gate_detector.config['output_shape'], 
                                                         0.7, ret=True)
    
        cv2.imshow("drone", pred_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        
    cv2.destroyAllWindows()
