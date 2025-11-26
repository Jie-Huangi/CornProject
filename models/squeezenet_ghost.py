import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ["SqueezeNet", "squeezenet1_1"]

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=1):
        super(GhostModule, self).__init__()
        self.ratio = ratio
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // ratio, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels // ratio),
            nn.ReLU(inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(out_channels // ratio, out_channels // ratio, dw_size, 1, padding=dw_size//2, groups=out_channels//ratio, bias=False),
            nn. BatchNorm2d(out_channels // ratio),
            nn. ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self. expand1x1 = nn. Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = GhostModule(squeeze_planes, expand3x3_planes, kernel_size=3, ratio=2, dw_size=3)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3(x)], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, groups=3),  # 深度卷积
            nn.Conv2d(3, 64, kernel_size=1),   # 逐点卷积
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
        )

        final_conv = nn.Conv2d(384, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn. Dropout(p=dropout), final_conv, nn.PReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

def squeezenet1_1(**kwargs):
    return SqueezeNet(**kwargs)