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
        self.branch2=nn.Sequential(
            BasicConv2d(in_channels, filter33reduce, kernel_size=1),
            BasicConv2d(filter33, kernel_size=1, padding=1)
        )
        self.branch3=nn.Sequential(
            BasicConv2d(in_channels, filter55reduce, kernel_size=1, padding=1),
            BasicConv2d(filter55reduce, filter55, kernel_size=5, padding=2)
        )
        self.branch4=nn.Sequential(
            nn.MaxPooling2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        outputs=[branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)



class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool=nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv=BasicConv2d(in_channels, 128, kernel_size=1)
        self.FC1=nn.Linear(2048, 1024)
        self.FC2=nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x=self.averagePool(x)
        x=self.conv(x)
        x=torch.flatten(x, 1)
        x=F.dropout(x, 0.5, training=self.training)
        x=F.relu(self.FC1(x), inplace=True)
        x=F.dropout(x, 0.5, training=self.training)
        x=self.FC2(x)
        return x



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
        if self.aux_logits:
            self.aux1=InceptionAux(512, num_classes)
            self.aux2=InceptionAux(528, num_classes)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1)) #specify pooling size (Height,Wide)
        self.dropout=nn.Dropout(0.4)
        self.FC=nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.maxpool2(x)
        x=self.inception3a(x)
        x=self.inception3b(x)
        x=self.maxpool3(x)
        x=self.inception4a(x)
        if self.training and self.aux_logits:
            aux1=self.aux1(x)
        x=self.inception4b(x)
        x=self.inception4c(x)
        x=self.inception4d(x)
        if self.training and self.aux_logits:
            aux2=self.aux2(x)
        x=self.inception4e(x)
        x=self.maxpool4(x)
        x=self.inception5a(x)
        x=self.inception5b(x)
        x=self.avgpool(x)
        x=torch.flatten(x, 1)
        x=self.dropout(x)
        x=self.FC(x)
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


GoogLeNet=GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
GoogLeNet=GoogLeNet.to(device)
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(GoogLeNet.parameters(), lr=0.0003)