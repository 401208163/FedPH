# -*- coding: utf-8 -*-
"""
@author: Kuang Hangdong
@software: PyCharm
@file: TestPath.py
@time: 2023/3/3 10:19
"""
import glob
import random
import platform
import numpy as np
from PIL import Image
import torch


class args_parser():
    def __init__(self):
        # Federated Learning Arguments
        self.rounds = 50
        self.num_users = 5
        self.alg = 'fedph'  # local, fedph, fedproto, fedavg, fedprox
        self.train_ep = 1
        self.local_bs = 8
        self.lr = 0.001
        self.momentum = 0.5
        self.weight_decay = 1e-4
        self.optimizer = 'sgd'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.seed = 420

        # Model Arguments
        self.num_bb = 1

        # Data Arguments
        self.dataset = 'vehicles'
        self.train_size = 0.8
        self.num_classes = 6
        self.alpha = 1
        self.non_iid = 1

        # Loss Funtion
        self.dis = 'cos'
        self.ld = 1  # fedproto fedph
        self.mu = 1  # fedprox
        self.temperature = 0.5  # fedph

        # Data
        self.classes = [
            'Bus',
            'Microbus',
            'Minivan',
            'Sedan',
            'SUV',
            'Truck'
        ]
        self.domains = ['Sunny',
                        'Rainy',
                        'Snowy',
                        'Fog',
                        'Cloudy'
                        ]
        self.file_path = './data/VehicleDataset-DallE'


args = args_parser()

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


def split_data(domains, classes, file_path):
    image_paths = np.array([])

    for domain in domains:
        for cla in classes:
            image_paths = np.append(image_paths, np.array(glob.glob(f'{file_path}/{cla}/{domain}/*.jpg')))
    np.random.shuffle(image_paths)

    print(len(image_paths))

    vehicles_to_index = {cls: index for index, cls in enumerate(classes)}
    labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-3]), image_paths)))
    client_indexs = dirichlet_split_noniid(labels, args.alpha, args.num_users)
    return image_paths, labels, client_indexs

split_data(args.domains,args.classes,args.file_path)