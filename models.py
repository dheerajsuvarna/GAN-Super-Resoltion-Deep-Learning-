# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

TODO:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


class resnet_generator(nn.Module):
    def __init__(self):
        super(resnet_generator, self).__init__()

        resnet18 = models.resnet18(pretrained=True)

        for param in resnet18.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), bias=False)
        nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.layer1 = resnet18.layer1

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv2.weight)
        self.bb1 = resnet18.layer2[1]

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv3.weight)
        self.bb2 = resnet18.layer3[1]

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv4.weight)
        self.bb3 = resnet18.layer4[1]

        self.conv5 = nn.Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        nn.init.xavier_normal(self.conv5.weight)
        self.bn2 = nn.BatchNorm2d(64)

        self.upsampler = upsampleBlock_modified(64, 256)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        nn.init.xavier_normal(self.conv6.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        y = x.clone()
        x = self.layer1(x)
        x = self.bb1(self.conv2(x))
        x = self.bb2(self.conv3(x))
        x = self.bb3(self.conv4(x))
        x = self.bn2(self.conv5(x)) + y
        x = self.upsampler(x)
        return self.conv6(x)


class upsampleBlock_modified(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock_modified, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.conv.weight)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))


class DiscriminatorPretrained(nn.Module):
    def __init__(self, imgSize):
        super(DiscriminatorPretrained, self).__init__()
        self.vgg = models.vgg13_bn(pretrained=True)

        for param in self.vgg.parameters():
            param.requires_grad = False

        del self.vgg.classifier

        self.fc1 = nn.Linear(int(imgSize * imgSize /2), 1024)
        nn.init.xavier_normal(self.fc1.weight)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1)
        nn.init.xavier_normal(self.fc2.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # for i in range(5):
        #    x = self.__getattr__('conv1' + str(i))(x)
        #    x = self.__getattr__('bn1' + str(i))(x)
        #    x = self.__getattr__('relu1' + str(i))(x)
        #    x = self.__getattr__('conv2' + str(i))(x)
        #    x = self.__getattr__('bn2' + str(i))(x)
        #    x = self.__getattr__('relu2' + str(i))(x)
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(self.leaky(self.fc1(x)))
        return self.sigmoid(x)

def swish(x):
    return x * F.sigmoid(x)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))

class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(int(self.upsample_factor/2)):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(self.upsample_factor/2)):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
