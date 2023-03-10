# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 19:54
# @Author  : Kuang Hangdong
# @File    : privacy.py
# @Software: PyCharm
# @desc    :
import torch
import torch.nn.functional as F
import numpy as np


def add_noise_proto(device, local_protos, scale, noise_type, threshold, is_not_the):
    """
    differential privacy
    :param device:
    :param local_protos:
    :param scale:
    :param noise_type:
    :param threshold:
    :param is_not_the:
    :return:
    """
    scale_scalar = scale
    for label in local_protos.keys():
        local_protos[label] = F.normalize(local_protos[label], dim=0)
        scale = torch.full(size=local_protos[label].shape, fill_value=scale_scalar, dtype=torch.float32)
        if noise_type == "gaussian":
            dist = torch.distributions.normal.Normal(0, scale)
        elif noise_type == "laplacian":
            dist = torch.distributions.laplace.Laplace(0, scale)
        elif noise_type == "exponential":
            rate = 1 / scale
            dist = torch.distributions.exponential.Exponential(rate)
        else:
            dist = torch.distributions.normal.Normal(0, scale)

        noise = dist.sample().to(device)
        if is_not_the:
            local_protos[label] = local_protos[label] + noise / (threshold - 1)
        else:
            local_protos[label] = local_protos[label] + noise

    return local_protos


def THE_Encryption(protos, PublicKey):
    """
    the threshold scheme
    :param protos:
    :param PublicKey:
    :return:
    """
    protos_ = {}
    for label, proto in protos.items():
        protos_[label] = np.array([PublicKey.encrypt(number) for number in proto.numpy().tolist()])
    return protos_


def THE_Decryption(device, global_avg_protos, Key, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta):
    """
    the threshold scheme
    :param device:
    :param global_avg_protos:
    :param Key:
    :param NUMBER_PLAYERS:
    :param CORRUPTION_THRESHOLD:
    :param PublicKey:
    :param SecretKeyShares:
    :param theta:
    :return:
    """
    global_avg_protos_ = {}
    for label, protos in global_avg_protos.items():
        protos_ = [Key.decrypt(proto, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta) for proto
                   in protos]
        protos_ = torch.tensor(list(map(float, protos_))).to(device)
        global_avg_protos_[label] = protos_
    return global_avg_protos_
