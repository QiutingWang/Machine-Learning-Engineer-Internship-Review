## AlexNet PyTorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchversion.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data
import warnings
warnings.simplefilter(action='ignore')
device='cuda' if torch.cuda.is_available() else 'cpu'

#define parameters
n_epochs=90
batch_size=128
momentum=0.9

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,stride=4, padding=2)
        self.relu=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2=nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool2=nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3=nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4=nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5=nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool5=nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten=nn.Flatten()
        self.FC6=nn.Linear(6*6*256, 4096)
        self.FC7=nn.Linear(4096, 4096)
        self.FC8=nn.Linear(4096, 1000)
        self.FC9=nn.Linear(1000, 2)
    
    def forward(self, x):
        x=self.ReLU(self.conv1(x))
        x=self.pool1(x)
        x=self.ReLU(self.conv2(x))
        x=self.pool2(x)
        x=self.ReLU(self.conv3(x))
        x=self.ReLU(self.conv4(x))
        x=self.ReLU(self.conv5(x))
        x=self.pool5(x)
        x=self.flatten(x)
        x=self.FC6(x)
        x=F.dropout(x, p=0.5)
        x=self.FC7(x)
        x=F.dropout(x, p=0.5)
        x=self.FC8(x)
        x=F.dropout(x, p=0.5)
        x=self.FC9(x)
        return x

AlexNet = AlexNet.to(device)  
optimizer=optim.SGD(AlexNet.parameters(), lr=0.001, momentum=momentum)
loss_function=nn.CrossEntropyLoss()



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

model=Sequential()

model.add(Conv2D(filter=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filter=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(4096, input_shape=(6,6,256)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()
