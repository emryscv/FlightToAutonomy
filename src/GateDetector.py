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
