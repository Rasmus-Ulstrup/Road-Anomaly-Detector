import torch
import torch.nn as nn
import torch.nn.functional as F


class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),  # Change 3 to 1 here
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 5
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Side outputs
        self.side1 = nn.Conv2d(128, 1, kernel_size=1)
        self.side2 = nn.Conv2d(256, 1, kernel_size=1)
        self.side3 = nn.Conv2d(512, 1, kernel_size=1)
        self.side4 = nn.Conv2d(1024, 1, kernel_size=1)
        self.side5 = nn.Conv2d(1024, 1, kernel_size=1)
        
        # Merge layer
        self.merge = nn.Conv2d(5, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()
    def forward(self, x):
        # Forward through layers
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        
        # Side outputs with sigmoid activation
        # side1 = torch.sigmoid(F.interpolate(self.side1(out1), size=x.shape[2:], mode='bilinear', align_corners=False))
        # side2 = torch.sigmoid(F.interpolate(self.side2(out2), size=x.shape[2:], mode='bilinear', align_corners=False))
        # side3 = torch.sigmoid(F.interpolate(self.side3(out3), size=x.shape[2:], mode='bilinear', align_corners=False))
        # side4 = torch.sigmoid(F.interpolate(self.side4(out4), size=x.shape[2:], mode='bilinear', align_corners=False))
        # side5 = torch.sigmoid(F.interpolate(self.side5(out5), size=x.shape[2:], mode='bilinear', align_corners=False))
        side1 = F.interpolate(self.side1(out1), size=x.shape[2:], mode='bilinear', align_corners=False)
        side2 = F.interpolate(self.side2(out2), size=x.shape[2:], mode='bilinear', align_corners=False)
        side3 = F.interpolate(self.side3(out3), size=x.shape[2:], mode='bilinear', align_corners=False)
        side4 = F.interpolate(self.side4(out4), size=x.shape[2:], mode='bilinear', align_corners=False)
        side5 = F.interpolate(self.side5(out5), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Merge outputs
        merged = self.merge(torch.cat([side1, side2, side3, side4, side5], dim=1))
        final_output = self.final_activation(merged)
        return [side1, side2, side3, side4, side5, final_output]
        # return final_output
#model = HED()