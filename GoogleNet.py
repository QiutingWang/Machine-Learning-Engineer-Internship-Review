import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu=nn.ReLU(inplace=True)

    def forward(self):
        x=self.conv(x)
        x=self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, filter11, filter33reduce, filter33, filter55reduce, filter55, pool_proj):
        self.branch1=BasicConv2d(in_channels, filter11, kernel_size=1)



class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits=aux_logits

        self.conv1=BasicConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2=BasicConv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3=BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception3a=Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b=Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception4a=Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b=Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c=Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d=Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e=Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception5a=Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b=Inception(832, 394, 192, 384, 48, 128, 128)