import torch
import torch.nn as nn
import torch.nn.functional as F
class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()

        # Encoder
        self.encoder1 = self._conv_block(1, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)
        self.encoder5 = self._conv_block(512, 1024, pool=False)  # Remove pooling from the last layer
        
        # Decoder
        self.upconv5 = self._upconv_block(1024, 512)
        self.decoder5 = self._conv_block(1024, 512, pool=False)
        self.upconv4 = self._upconv_block(512, 256)
        self.decoder4 = self._conv_block(512, 256, pool=False)
        self.upconv3 = self._upconv_block(256, 128)
        self.decoder3 = self._conv_block(256, 128, pool=False)
        self.upconv2 = self._upconv_block(128, 64)
        self.decoder2 = self._conv_block(128, 64, pool=False)
        
        # Final output layer
        self.output_conv = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
        self.output_activation = nn.Sigmoid()
        
    def _conv_block(self, in_channels, out_channels, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        return nn.Sequential(*layers)
    
    def _upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder path with interpolation to ensure matching dimensions
        d5 = self.upconv5(e5)
        e4_resized = F.interpolate(e4, size=d5.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat((d5, e4_resized), dim=1)
        d5 = self.decoder5(d5)
        
        d4 = self.upconv4(d5)
        e3_resized = F.interpolate(e3, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat((d4, e3_resized), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        e2_resized = F.interpolate(e2, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat((d3, e2_resized), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        e1_resized = F.interpolate(e1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat((d2, e1_resized), dim=1)
        d2 = self.decoder2(d2)
        
        # Output layer
        out = self.output_conv(d2)
        out = self.output_activation(out)
        
        return out
#model = unet().cuda()
# print(model)