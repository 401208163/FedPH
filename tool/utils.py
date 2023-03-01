# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 18:07
# @Author  : Kuang Hangdong
# @File    : utils.py
# @Software: PyCharm
# @desc    :
import glob
import random
import platform
import numpy as np
from PIL import Image

import torch
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.55, 0.58, 0.6], std=[0.23, 0.23, 0.25]),
])


def read_path(classes, domains, filename):

    paths_dict = {domain: [] for domain in domains}

    for cla in classes:
        for domain in domains:
            paths_dict[domain] = paths_dict[domain] + glob.glob(f'./data/{filename}/{cla}/{domain}/*.jpg')

    for domain in domains:
        random.shuffle(paths_dict[domain])

    classes_to_index = {cls: index for index, cls in enumerate(classes)}
    if platform.system() == "Windows":
        labels_dict = {domain: list(map(lambda path: classes_to_index.get(path.split('/')[-2]), paths_dict[domain])) for
                       domain in domains}
    else:
        labels_dict = {domain: list(map(lambda path: classes_to_index.get(path.split('/')[-3]), paths_dict[domain])) for
                       domain in domains}
    return paths_dict, labels_dict


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        pillow_image = Image.open(image_path)
        pillow_image = pillow_image.convert('RGB')
        image = transform(pillow_image)

        label = self.labels[index]

        return image, label

    def __len__(self):
        return len(self.image_paths)


def criterion_CL(global_protos, features, labels, distance, args):
    # pos
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    global_protos_ = torch.cat(list(map(lambda c: global_protos[c].reshape(1, -1), range(args.num_classes))), dim=0)[
        labels]
    posi = distance(features, global_protos_)
    logits = posi.reshape(-1, 1)

    # neg
    key_masks = np.array(np.eye(args.num_classes), dtype=bool)
    value_masks = ~key_masks
    c = np.arange(args.num_classes)
    kv = {c[key_mask][0]: c[value_mask] for key_mask, value_mask in zip(list(key_masks), list(value_masks))}
    masks = torch.tensor(np.array(list(map(lambda label: list(kv.get(label)), labels.cpu().numpy()))))
    for i in range(args.num_classes - 1):
        global_protos_ = torch.cat(list(map(lambda c: global_protos[c].reshape(1, -1), range(args.num_classes))), dim=0)
        nega = distance(features, global_protos_[masks[:, i]])
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

    # loss
    logits /= args.temperature
    targets = torch.zeros(labels.size(0)).cuda().long()
    loss = args.ld * criterion(logits, targets)
    return loss


def agg_func(protos):
    """
    Average the protos for each local user
    """
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = torch.stack(proto_list)
        agg_protos[label] = torch.mean(proto, dim=0).data

    return agg_protos


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label
