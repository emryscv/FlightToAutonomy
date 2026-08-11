import cv2
from src.GateDetector import GateDetector
import utils_io
import numpy as np

gate_detector = GateDetector()

img = cv2.imread("./testImages/UZHGate.jpg")
prediction = gate_detector.predict(img)

pred_img, bboxes = utils_io.display_target_woWH(np.float32(prediction[0]), 
                                                         img, 
                                                         gate_detector.config['output_shape'], 
        
                                                         0.7, ret=True)
    
cv2.imshow("drone", pred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()