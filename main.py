#opencv needs to go first than djitellopy, otherwise imshow wont work
import cv2 
from djitellopy import Tello
from time import sleep
import os
import subprocess
import math
import threading

def telemetry_thread(tello):
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

        cv2.imshow("drone", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

tello = Tello()
tello.connect()

tello.streamon()
sleep(1)

frame_read = tello.get_frame_read()

threading.Thread(target=telemetry_thread, args=(tello,), daemon=True).start()

tello.takeoff()

tello.move_up(50)
tello.send_rc_control(0, 50, 0, 0)
sleep(3)
tello.send_rc_control(0, 0, 0, 0)
sleep(2)
tello.send_rc_control(0, -50, 0, 0)
sleep(2)
tello.send_rc_control(0, 0, 0, 0)

tello.land()
tello.streamoff()
cv2.destroyAllWindows()