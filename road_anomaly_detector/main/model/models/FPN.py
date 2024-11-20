import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=5, activation=nn.ReLU(inplace=True)):
        super(FPN, self).__init__()

        # Define the initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = activation

        # Define the encoder blocks
        self.encoders = nn.ModuleList()
        for i in range(num_blocks):
            self.encoders.append(self._conv_block(in_channels=64 * 2**i, out_channels=64 * 2**(i+1)))

        # Define the decoder blocks (can be modified for different architectures)
        self.decoders = nn.ModuleList()
        for i in range(num_blocks-1, -1, -1):
            if i == num_blocks-1:  # No pooling in the last decoder
                self.decoders.append(self._upconv_block(in_channels=64 * 2**(i+1), out_channels=64 * 2**i))
            else:
                self.decoders.append(self._upconv_block(in_channels=64 * 2**(i+1), out_channels=64 * 2**i))
            self.decoders.append(self._conv_block(in_channels=64 * 2**(i+1), out_channels=64 * 2**i, pool=False))

        # Define the final output layer (can be modified for different tasks)
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=(1, 1), stride=(1, 1))

    def _conv_block(self, in_channels, out_channels, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
            self.act1,
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
            self.act1,
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        return nn.Sequential(*layers)

    def _upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        # Encode features
        encoded_features = []
        x = self.act1(self.bn1(self.conv1(x)))
        for encoder in self.encoders:
            x = encoder(x)
            encoded_features.append(x)

        # Decode features
        for i in range(len(self.decoders)):
            if i % 2 == 0:  # Upconv
                x = self.decoders[i](x)
                if len(encoded_features) > 0:
                    # Concatenate with features from the corresponding encoder level
                    e = encoded_features.pop()
                    x = torch.cat((x, F.interpolate(e, size=x.shape[2:], mode='bilinear', align_corners=False)), dim=1)
            else:  # Regular convolution
                x = self.decoders[i](x)

        # Final output
        out = self.output_conv(x)

        return out
