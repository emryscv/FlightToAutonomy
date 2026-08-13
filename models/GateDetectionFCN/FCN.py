import torch.nn as nn
import torch
import numpy as np

class FCN(nn.Module):
    def __init__(self, model, upsample_mode):
        super(FCN, self).__init__()
        self.vgg = model
        self.upsample_mode = upsample_mode
        self.num_classes = 1

        assert self.upsample_mode in [
            "32s",
            "16s",
            "8s",
        ], "Invalid upsampling: only support '32s', '16s' or '8s'"

        # upsampling layers
        self.unconv8x = nn.ConvTranspose2d(
            self.num_classes,
            self.num_classes,
            kernel_size=16,
            stride=8,
            padding=4,
            bias=False,
        )
        self.unconv16x = nn.ConvTranspose2d(
            self.num_classes,
            self.num_classes,
            kernel_size=32,
            stride=16,
            padding=8,
            bias=False,
        )
        self.unconv32x = nn.ConvTranspose2d(
            self.num_classes,
            self.num_classes,
            kernel_size=64,
            stride=32,
            padding=16,
            bias=False,
        )
        self.unconv2x_pool3 = nn.ConvTranspose2d(
            self.num_classes,
            self.num_classes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.unconv2x_pool5 = nn.ConvTranspose2d(
            self.num_classes,
            self.num_classes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        # 1x1 convolution layers for dimension/channel reduction
        self.linear_conv_pool3 = nn.Conv2d(256, 1, kernel_size=1)
        self.linear_conv_pool4 = nn.Conv2d(512, 1, kernel_size=1)
        self.linear_conv_pool5 = nn.Conv2d(512, 1, kernel_size=1)

        # initializing the transposed convolution layer with bilinear interpolation, and make it learnable
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(
                    self.get_upsampling_weight(
                        m.in_channels, m.out_channels, m.kernel_size[0]
                    )
                )

    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        # make a 2D bilinear kernel suitable for upsampling
        factor = (kernel_size + 1) // 2
        center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

        weight = np.zeros(
            (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64
        )
        weight[range(in_channels), range(out_channels), :, :] = filt

        return torch.from_numpy(weight).float()

    def forward(self, x):
        pool2 = self.vgg.blocks[0](x)  # block[1] has 2 max-pooling layers
        pool3 = self.vgg.blocks[1](pool2)
        pool4 = self.vgg.blocks[2](pool3)
        pool5 = self.vgg.blocks[3](pool4)

        if self.upsample_mode == "32s":
            out = self.unconv32x(self.linear_conv_pool5(pool5))

        elif self.upsample_mode == "16s":
            upsample_2x = self.unconv2x_pool5(
                self.linear_conv_pool5(pool5)
            ) + self.linear_conv_pool4(pool4)
            out = self.unconv16x(upsample_2x)

        elif self.upsample_mode == "8s":
            upsample_2x = self.unconv2x_pool5(
                self.linear_conv_pool5(pool5)
            ) + self.linear_conv_pool4(pool4)
            upsample_4x = self.unconv2x_pool3(upsample_2x) + self.linear_conv_pool3(
                pool3
            )
            out = self.unconv8x(upsample_4x)

        return out
