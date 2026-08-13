import torch
from models.GateDetectionFCN.MiniVGG import MiniVGG
from models.GateDetectionFCN.FCN import FCN
import torch.nn as nn
import os


class GateDetection:
    def __init__(
        self,
        vgg_config,
        upsample_mode,
        batch_norm,
        num_epoch,
        #train_set,
        #test_set,
        #train_loader,
        #test_loader,
        pretrained=False,
    ):
        self.vgg_config = vgg_config
        self.upsample_mode = upsample_mode
        self.batch_norm = batch_norm
        self.num_epoch = num_epoch
        #self.train_set = train_set
        #self.test_set = test_set
        #self.train_loader = train_loader
        #self.test_loader = test_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mini_vgg = MiniVGG(self.vgg_config, self.batch_norm)
        self.fcn_model = FCN(self.mini_vgg, self.upsample_mode).to(self.device)
        # binary cross entropy loss function
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self.pretrained_model_path = "models/GateDetectionFCN/weights/{}_{}_bn_{}".format(
            self.vgg_config, self.upsample_mode, str(self.batch_norm)
        )

        if pretrained:
            self.fcn_model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.pretrained_model_path, "epoch_{}.pt".format(self.num_epoch)
                    ),
                    map_location=self.device,
                )
            )

    # def train(self):
    #     """
    #     Train the FCN model with given structure.
    #     After training, save trained model & logger for loss.
    #     """
    #     # loss logger and pretrained model directory
    #     logger_train = open(
    #         PROJECT_PATH
    #         + "logs/loss_{}_{}_bn_{}.txt".format(
    #             self.vgg_config, self.upsample_mode, str(self.batch_norm)
    #         ),
    #         "w+",
    #     )
    #     if not os.path.exists(self.pretrained_model_path):
    #         os.mkdir(self.pretrained_model_path)

    #     optimizer = torch.optim.Adam(self.fcn_model.parameters(), lr=1e-4)

    #     # training process
    #     start_time = time.time()
    #     for epoch_idx in range(self.num_epoch + 1):
    #         for iter, batch in enumerate(self.train_loader):
    #             optimizer.zero_grad()
    #             inputs = batch["img"].to(self.device)
    #             labels = batch["mask"].to(self.device)
    #             loss = self.criterion(self.fcn_model(inputs), labels)
    #             loss.backward()
    #             optimizer.step()

    #         # print and store loss for each epoch
    #         print("epoch{}, loss: {}".format(epoch_idx, loss.item()))
    #         logger_train.write("{}, {}\n".format(epoch_idx, loss.item()))
    #         logger_train.flush()

    #         # save trained model every 200 epochs
    #         if epoch_idx % 200 == 0:
    #             torch.save(
    #                 self.fcn_model.state_dict(),
    #                 os.path.join(
    #                     self.pretrained_model_path, "epoch_{}.pt".format(epoch_idx)
    #                 ),
    #             )
    #     end_time = time.time()

    #     logger_train.close()
    #     print("Finished training, took %.2f seconds\n" % (end_time - start_time))

    # def demo_show(self, apply_sigmoid=False):
    #     """
    #     Demonstrate the original image & ground truth & prediction
    #     on several samples randomly sampled from training/test datasets.
    #     """
    #     # plot images and masks
    #     num_of_subs = 6
    #     train_seed = np.random.randint(
    #         low=1, high=len(train_set), size=int(num_of_subs / 2)
    #     )
    #     test_seed = np.random.randint(
    #         low=1, high=len(test_set), size=int(num_of_subs / 2)
    #     )
    #     sample_seed = np.concatenate((train_seed, test_seed))

    #     fig = plt.figure(dpi=200)
    #     fig.tight_layout()
    #     plt.subplots_adjust(bottom=0.0, top=0.6)

    #     for id, i in enumerate(sample_seed):
    #         train_or_test = "train" if id < 3 else "test"
    #         sample = self.train_set[i] if id < 3 else self.test_set[i]

    #         # show img
    #         ax1 = plt.subplot(3, num_of_subs, id + 1)
    #         ax1.set_title("{} img #{}".format(train_or_test, i), fontsize=4)
    #         ax1.axis("off")
    #         ax1.imshow(sample["img"].numpy().transpose(1, 2, 0))

    #         # show mask
    #         ax2 = plt.subplot(3, num_of_subs, id + num_of_subs + 1)
    #         ax2.set_title("{} mask #{}".format(train_or_test, i), fontsize=4)
    #         ax2.axis("off")
    #         ax2.imshow(sample["mask"].numpy().squeeze(0), cmap="gist_gray")

    #         # show output
    #         ax3 = plt.subplot(3, num_of_subs, id + 2 * num_of_subs + 1)
    #         ax3.set_title("{} prediction #{}".format(train_or_test, i), fontsize=4)
    #         ax3.axis("off")

    #         output = self.fcn_model(sample["img"].to(self.device).unsqueeze(0))
    #         if apply_sigmoid == True:
    #             output = torch.sigmoid(output)

    #         # host tensor memory on cpu before transfer to cuda
    #         ax3.imshow(output.detach().cpu().numpy().reshape(288, 288))

    # def compute_IoU_PixelAccuracy(self):
    #     """
    #     Calculate the mean Intersaction of Unions (IoU) and Pixel Accuracy of given model on test dataset.
    #     Mean IoU: true_positive / (true_positive + false_positive + false_negative).
    #     Pixel Acc.: (number of pixels of class i predicted to belong to class i) / (total number of pixels of class i).
    #     """
    #     iou_score = 0
    #     pixel_acc = 0

    #     for iter, batch in enumerate(self.test_loader):
    #         inputs = batch["img"].to(self.device)
    #         labels = batch["mask"].to(self.device)

    #         logits = self.fcn_model(inputs)
    #         y_pred = torch.sigmoid(logits).detach().cpu().numpy()
    #         y_true = labels.cpu().numpy()

    #         intersection = np.logical_and(np.rint(y_true), np.rint(y_pred))
    #         union = np.logical_or(np.rint(y_true), np.rint(y_pred))
    #         iou_score += np.sum(intersection) / np.sum(union)
    #         pixel_acc += np.sum(intersection) / np.sum(np.rint(y_true))

    #     return {"IoU": iou_score / len(test_loader), "PA": pixel_acc / len(test_loader)}

    # def inference_speed(self):
    #     """
    #     Evaluate the mean inference speed of the model on GPU over the training dataset.
    #     """
    #     total_time = 0
    #     for i in range(len(self.test_set)):
    #         sample = self.test_set[i]
    #         start_eval = time.time()
    #         output = self.fcn_model(sample["img"].to(self.device).unsqueeze(0))
    #         end_eval = time.time()
    #         total_time += end_eval - start_eval

    #     # inference speed is evaluated in miliseconds
    #     return 1000 * total_time / len(self.test_set)


