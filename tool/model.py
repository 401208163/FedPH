# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 18:14
# @Author  : Kuang Hangdong
# @File    : model.py
# @Software: PyCharm
# @desc    :
import torch
import torchvision

def if_not_backbone(args):
    if args.backbone == "resnet":
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        backbone = torch.nn.Sequential(*modules)
        backbone = backbone.to(args.device)
        backbone = backbone.eval()
        backbone_list = [backbone]
        return backbone_list

class Model(torch.nn.Module):
    def __init__(self, in_d, out_d, num_classes):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(in_d, out_d)
        self.fc2 = torch.nn.Linear(out_d, num_classes)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x.squeeze()))
        x = torch.nn.functional.dropout(x1, training=self.training)
        x = self.fc2(x)
        return x, x1