import cv2
from torchvision import transforms

from models.GateDetectionFCN import GateDetection
# import json
# import numpy as np

# from models.GateNet.network import network_architecture

# class GateDetector:
#     def __init__(self):
#         # 1) Load config
#         with open("./models/GateNet/configs.json", "r") as f:
#             self.config = json.load(f)

#         # 2) Build architecture
#         self.model = network_architecture(self.config)

#         # 3) Load checkpoint weights
#         weights_path = "./models/GateNet/weights/model_2022-01-24-01-06_-500.h5"
#         self.model.load_weights(weights_path)

#     def predict(self, img):
#         x = cv2.resize(img, (self.config["input_shape"][1], self.config["input_shape"][0]))
#         x = x.astype(np.float32)/255.0

#         x = np.reshape(x, (1, self.config["input_shape"][0], self.config["input_shape"][1], self.config["input_shape"][2]))

#         y = self.model.predict(x)
        
        
#         print("input shape:", x.shape)     # expected: (N, 120, 160, 3)
#         print("output shape:", y.shape)    # expected: (N, 3, 4, 5)

#         print("output values:", y)

#         return y



def gate_detect_init(vgg_config, upsample_mode, batch_norm, print_summary=False, pretrained=True):
    return GateDetection(vgg_config=vgg_config, upsample_mode=upsample_mode,
                                batch_norm=batch_norm, num_epoch=1000,
                                # train_set=train_set, test_set=test_set,
                                # train_loader=train_loader, test_loader=test_loader,
                                print_summary=print_summary, pretrained=pretrained)

gate_detection = gate_detect_init('vgg-11', '8s', False, print_summary=True)
#gate_detection.train()
#gate_detection.inference_speed()
gate_detection.demo_show(apply_sigmoid=False)


gate_detection = gate_detect_init('vgg-5', '8s', False)


img = cv2.imread("../test_images/UZHGateSquare.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (288, 288))

x = transforms.ToTensor()(img).to(gate_detection.device).unsqueeze(0)

y = gate_detection.fcn_model(x)

cv2.imshow("output", y.detach().cpu().numpy().reshape((288, 288)))
cv2.waitKey(0)
cv2.destroyAllWindows()