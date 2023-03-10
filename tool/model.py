# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 18:14
# @Author  : Kuang Hangdong
# @File    : model.py
# @Software: PyCharm
# @desc    :
import torch
import torchvision
from torchvision import models
import torch.nn.functional as F

def if_not_backbone(device, num_bb):
    """
    :param device:
    :param num_bb:
    :return: backbone_list
    """
    backbone_list = [torchvision.models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval(),
                     torchvision.models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device).eval(),
                     torchvision.models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()]

    for i in range(len(backbone_list) - num_bb):
        backbone_list.pop()

    return backbone_list

class Model(torch.nn.Module):
    def __init__(self, in_d, out_d, num_classes):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(in_d, out_d)
        self.fc2 = torch.nn.Linear(out_d, num_classes)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x.squeeze()))
        x1 = F.normalize(x1, dim=1)

        x2 = self.fc2(x1)
        return x2, x1