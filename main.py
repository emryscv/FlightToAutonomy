#opencv needs to go first than djitellopy, otherwise imshow wont work

import cv2 
from djitellopy import Tello
from time import sleep
import threading
from src.telemetry_thread import telemetry_thread

tello = Tello()
tello.connect()

tello.streamon()
sleep(1)

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
