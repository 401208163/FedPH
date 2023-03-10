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
import torch.utils.data

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
])


def split_data(classes, domains, train_size, file_path):
    """
    splitting datasets
    :param classes:
    :param domains:
    :param train_size:
    :param file_path:
    :return:
    """
    image_paths = np.array([])

    for domain in domains:
        for cla in classes:
            image_paths = np.append(image_paths, np.array(glob.glob(f'{file_path}/{cla}/{domain}/*.jpg')))
    np.random.shuffle(image_paths)
    length = int(train_size * len(image_paths))
    train_image_paths = image_paths[:length]
    test_image_paths = image_paths[length:]

    vehicles_to_index = {cls: index for index, cls in enumerate(classes)}
    if platform.system() == "Windows":
        train_labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-2]), train_image_paths)))
        test_labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-2]), test_image_paths)))
        return train_image_paths, train_labels, test_image_paths, test_labels
    elif platform.system() != "Windows":
        train_labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-3]), train_image_paths)))
        test_labels = np.array(list(map(lambda path: vehicles_to_index.get(path.split('/')[-3]), test_image_paths)))
        return train_image_paths, train_labels, test_image_paths, test_labels
    else:
        print("What is the current system?")


def dirichlet_split_noniid(train_labels, alpha, num_users):
    """
    dirichlet sampling
    :param train_labels:
    :param alpha:
    :param num_users:
    :return:
    """
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(num_users)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    client_idcs_ = []
    for client in client_idcs:
        np.random.shuffle(client)
        client_idcs_.append(client)

    return client_idcs_


def noniid(train_num_shards, train_num_imgs, num_users, train_shots_max, num_classes, dataset, n_list, k_list):
    """
    train dataset label shift
    :param train_num_shards:
    :param train_num_imgs:
    :param num_users:
    :param train_shots_max:
    :param num_classes:
    :param dataset:
    :param n_list:
    :param k_list:
    :return:
    """
    idx_shard = [i for i in range(train_num_shards)]
    dict_users = {}
    idxs = np.arange(train_num_shards * train_num_imgs)
    labels = dataset.labels

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = train_shots_max
        classes = random.sample(range(0, num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list


def noniid_lt(test_num_shards, test_num_imgs, num_users, test_dataset, classes_list):
    """
    test dataset label shift
    :param test_num_shards:
    :param test_num_imgs:
    :param num_users:
    :param test_dataset:
    :param classes_list:
    :return:
    """
    idx_shard = [i for i in range(test_num_shards)]
    dict_users = {}
    idxs = np.arange(test_num_shards * test_num_imgs)
    labels = test_dataset.labels
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    for i in range(num_users):
        k = 40
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * 40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data

    return dict_users


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
    """
    regular term
    :param global_protos:
    :param features:
    :param labels:
    :param distance_function:
    :param args:
    :return:
    """
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # pos
    protos_list = list(map(lambda cla: global_protos.get(cla).reshape(1, -1), range(args.num_classes)))
    # print(f"protos_list:{protos_list}")
    protos = torch.cat(protos_list, dim=0)
    # print(f"protos:{protos}")
    global_protos_ = protos[labels]
    # print(f"global_protos_:{global_protos_}")

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
    Average the protos
    :param protos:
    :return:
    """
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = torch.stack(proto_list)
        agg_protos[label] = torch.mean(proto, dim=0).data

    return agg_protos


def proto_aggregation(local_protos_list, is_not_the):
    """
    Protos aggregation
    :param local_protos_list:
    :param is_not_the:
    :return:
    """
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
            if is_not_the:
                proto = 0 * proto_list[0]
            else:
                proto = 0 * proto_list[0].data

            for i in proto_list:
                if is_not_the:
                    proto += i
                else:
                    proto += i.data

            agg_protos_label[label] = proto / len(proto_list)
        else:
            if is_not_the:
                agg_protos_label[label] = proto_list[0]
            else:
                agg_protos_label[label] = proto_list[0].data

    return agg_protos_label
