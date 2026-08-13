from collections import OrderedDict
import torch.nn as nn


class MiniVGG(nn.Module):
    def __init__(self, config, batch_norm, in_channels=3):
        super(MiniVGG, self).__init__()
        self.config = config
        self.bn_enabled = batch_norm
        self.in_channels = in_channels

        # initialize self.blocks as ModuleList so the blocks can be properly registered
        self.blocks = nn.ModuleList()
        assert self.config in [
            "vgg-5",
            "vgg-11",
        ], "Invalid config: only support 'vgg-5' or 'vgg-11'"

        if self.config == "vgg-5":
            self.net_arch = [[64, "M", 128, "M"], [256, "M"], [512, "M"], [512, "M"]]
        else:
            self.net_arch = [
                [64, "M", 128, "M"],
                [256, 256, "M"],
                [512, 512, "M"],
                [512, 512, "M"],
            ]

        for block in self.net_arch:
            # connect each layer inside each block
            self.layers = []
            for id, layer in enumerate(block):
                if layer == "M":
                    self.layers.append(
                        (
                            "maxpool({})".format(id),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                        )
                    )
                else:
                    self.layers.append(
                        (
                            "conv2d({})".format(id),
                            nn.Conv2d(
                                self.in_channels,
                                layer,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                            ),
                        )
                    )
                    self.layers.append(
                        ("relu({})".format(id), nn.ReLU(inplace=True))
                    )  # set inplace to save memory
                    if self.bn_enabled:
                        self.layers.append(("bn({})".format(id), nn.BatchNorm2d(layer)))
                    self.in_channels = layer

            # add to modulelist
            self.blocks.append(nn.Sequential(OrderedDict(self.layers)))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
