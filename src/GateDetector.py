import cv2
import json
import numpy as np

from models.GateNet.network import network_architecture

class GateDetector:
    def __init__(self):
        # 1) Load config
        with open("./models/GateNet/configs.json", "r") as f:
            self.config = json.load(f)

        # 2) Build architecture
        self.model = network_architecture(self.config)

        # 3) Load checkpoint weights
        weights_path = "./models/GateNet/weights/model_2022-01-24-01-06_-500.h5"
        self.model.load_weights(weights_path)

    def predict(self, img):
        x = cv2.resize(img, (self.config["input_shape"][1], self.config["input_shape"][0]))
        x = x.astype(np.float32)/255.0

        x = np.reshape(x, (1, self.config["input_shape"][0], self.config["input_shape"][1], self.config["input_shape"][2]))

        y = self.model.predict(x)
        
        
        print("input shape:", x.shape)     # expected: (N, 120, 160, 3)
        print("output shape:", y.shape)    # expected: (N, 3, 4, 5)

        print("output values:", y)

        return y

    def display_predicticed_gate_center(M, source_img, output_shape, threshold, ret=False):
        img = source_img.copy()
        img_height, img_width = img.shape[:2]

        bbox_results = []
        nrow, ncol = output_shape[0], output_shape[1]
        grid_dim_x = img_width/ncol
        grid_dim_y = img_height/nrow

        for i in range(nrow):
            for j in range(ncol):
                cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
                cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

                if M[i,j,0] > threshold:
                    cx, cy, distance, yaw_relative = M[i,j,1:]

                    #print(M[i,j,0], i, j, cx, cy, distance, yaw_relative)

                    cx_on_img = int(j * grid_dim_x + cx * grid_dim_x)
                    cy_on_img = int(i * grid_dim_y + cy * grid_dim_y)
                    cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,255,0), 3)

                    bbox_results.append((cx_on_img, cy_on_img, abs(float(distance)), yaw_relative))
                    
        #cv2.imwrite("display.png", img)
        if ret:
            return img, bbox_results

