# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 18:07
# @Author  : Kuang Hangdong
# @File    : utils.py
# @Software: PyCharm
# @desc    :
import os
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
])

"""
def read_path(classes, domains, filename):

    paths_dict = {domain: [] for domain in domains}

    # read data path
    for cls in classes:
        for domain in domains:
            paths_dict[domain] = paths_dict[domain] + glob.glob(f'./data/{filename}/{cls}/{domain}/*.jpg')

    # test data
    for domain in domains:
        for path in paths_dict[domain]:
            try:
                pillow_image = Image.open(path)
                pillow_image = pillow_image.convert('RGB')
            except:
                print("Exception data path:{path}")
                os.remove(path)

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
"""
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    client_idcs_ = []
    for client in client_idcs:
        np.random.shuffle(client)
        client_idcs_.append(client)

    return client_idcs_


def split_data(domains, classes, file_path, args):
    image_paths = np.array([])

    for domain in domains:
        for cla in classes:
            image_paths = np.append(image_paths, np.array(glob.glob(f'{file_path}/{cla}/{domain}/*.jpg')))
    np.random.shuffle(image_paths)
    length = int(args.train_size * len(image_paths))
    train_image_paths = image_paths[:length]
    test_image_paths = image_paths[length:]

    vehicles_to_index = {cls: index for index, cls in enumerate(classes)}
    if platform.system() == "Windows":
        train_labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-2]), train_image_paths)))
        test_labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-2]), test_image_paths)))
    else:
        train_labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-3]), train_image_paths)))
        test_labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-3]), test_image_paths)))
    client_indexs = dirichlet_split_noniid(train_labels, args.alpha, args.num_users)
    return train_image_paths, train_labels, test_image_paths, test_labels, client_indexs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        pillow_image = Image.open(image_path)
        pillow_image = pillow_image.convert('RGB')
        image = transform(pillow_image)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.image_paths)


def criterion_CL(global_protos, features, labels, distance_function, args):
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # pos
    global_protos_ = torch.cat(list(map(lambda c: global_protos[c].reshape(1, -1), range(args.num_classes))), dim=0)[labels]
    posi = distance_function(features, global_protos_)
    logits = posi.reshape(-1, 1)

    # neg
    key_masks = np.array(np.eye(args.num_classes), dtype=bool)
    value_masks = ~key_masks
    c = np.arange(args.num_classes)
    kv = {c[key_mask][0]: c[value_mask] for key_mask, value_mask in zip(list(key_masks), list(value_masks))}
    masks = torch.tensor(np.array(list(map(lambda label: list(kv.get(label)), labels.cpu().numpy())))).long()
    for i in range(args.num_classes - 1):
        global_protos_ = torch.cat(list(map(lambda c: global_protos[c].reshape(1, -1), range(args.num_classes))), dim=0)
        nega = distance_function(features, global_protos_[masks[:, i]])
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

    # loss
    logits /= args.temperature
    targets = torch.zeros(labels.size(0)).to(args.device).long()
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