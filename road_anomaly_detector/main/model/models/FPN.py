import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        # Encoder (Bottom-up pathway)
        self.encoder1 = self._conv_block(1, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)
        self.encoder5 = self._conv_block(512, 1024, pool=False)  # No pooling for the last block

        # Decoder (Top-down pathway)
        self.topdown4 = self._trans_conv_block(1024, 512)
        self.topdown3 = self._trans_conv_block(512, 256)
        self.topdown2 = self._trans_conv_block(256, 128)
        self.topdown1 = self._trans_conv_block(128, 64)

        # Lateral connections
        self.lateral5 = nn.Conv2d(1024, 512, kernel_size=(1, 1))  # Match encoder5 output to top-down pathway
        self.lateral4 = nn.Conv2d(512, 256, kernel_size=(1, 1))
        self.lateral3 = nn.Conv2d(256, 128, kernel_size=(1, 1))
        self.lateral2 = nn.Conv2d(128, 64, kernel_size=(1, 1))

        # Final output layers (side-outputs for multi-scale predictions)
        self.output4 = nn.Conv2d(512, 1, kernel_size=(1, 1))
        self.output3 = nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.output2 = nn.Conv2d(128, 1, kernel_size=(1, 1))
        self.output1 = nn.Conv2d(64, 1, kernel_size=(1, 1))

        # Merging step
        self.merge = nn.Conv2d(4, 1, kernel_size=(1, 1))
        self.final_activation = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        return nn.Sequential(*layers)

    def _trans_conv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        # Bottom-up pathway
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Top-down pathway with lateral connections
        td4 = self.topdown4(e5)
        td4 = F.interpolate(td4, size=e4.shape[2:], mode='bilinear', align_corners=False) + self.lateral5(e5)

        td3 = self.topdown3(td4)
        td3 = F.interpolate(td3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        lateral4_resized = F.interpolate(self.lateral4(e4), size=e3.shape[2:], mode='bilinear', align_corners=False)
        td3 = td3 + lateral4_resized

        td2 = self.topdown2(td3)
        td2 = F.interpolate(td2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        lateral3_resized = F.interpolate(self.lateral3(e3), size=e2.shape[2:], mode='bilinear', align_corners=False)
        td2 = td2 + lateral3_resized

        td1 = self.topdown1(td2)
        td1 = F.interpolate(td1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        lateral2_resized = F.interpolate(self.lateral2(e2), size=e1.shape[2:], mode='bilinear', align_corners=False)
        td1 = td1 + lateral2_resized

        # Multi-scale outputs
        # o4 = torch.sigmoid(F.interpolate(self.output4(td4), size=x.shape[2:], mode='bilinear', align_corners=False))
        # o3 = torch.sigmoid(F.interpolate(self.output3(td3), size=x.shape[2:], mode='bilinear', align_corners=False))
        # o2 = torch.sigmoid(F.interpolate(self.output2(td2), size=x.shape[2:], mode='bilinear', align_corners=False))
        # o1 = torch.sigmoid(F.interpolate(self.output1(td1), size=x.shape[2:], mode='bilinear', align_corners=False))
        o4 = F.interpolate(self.output4(td4), size=x.shape[2:], mode='bilinear', align_corners=False)
        o3 = F.interpolate(self.output3(td3), size=x.shape[2:], mode='bilinear', align_corners=False)
        o2 = F.interpolate(self.output2(td2), size=x.shape[2:], mode='bilinear', align_corners=False)
        o1 = F.interpolate(self.output1(td1), size=x.shape[2:], mode='bilinear', align_corners=False)

        # Merge outputs
        merged = torch.cat((o4, o3, o2, o1), dim=1)
        final_output = self.merge(merged)
        final_output = self.final_activation(final_output)

        return [o4, o3, o2, o1, final_output]
        #return final_output

# Instantiate and move model to GPU if available
# model = FPN().cuda()
# print(model)
