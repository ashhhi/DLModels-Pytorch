import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout, AdaptiveAvgPool2d
from torchvision import models

# 参考pytorch官方实现的VGG16
class VGG16_Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Conv2d(3, 64, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, 3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(64, 128, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(128, 128, 3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(128, 256, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, 3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(256, 512, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(512, 512, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            Linear(25088, 4096, bias=True),  # Linear层的in_feature参数需要根据输入图片大小手动计算出来
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(4096, 4096, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(4096, 1000, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 直接调用的预训练好的的VGG16，并更改classifier以适应自己的任务
class VGG16_Modified(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Test代码
if __name__ == '__main__':
    # VGG原论文使用224的输入大小
    fake_data = torch.ones(1, 3, 224, 224)
    MyVGG = VGG16_Original()
    print(MyVGG)
    res = MyVGG(fake_data)
    # print(res)
    print(res.shape)
