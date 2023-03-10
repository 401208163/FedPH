# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 16:38
# @Author  : Kuang Hangdong
# @File    : options.py
# @Software: PyCharm
# @desc    :
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=25, help='number of rounds of training')
    parser.add_argument('--num_users', type=int, default=5, help='number of users')
    parser.add_argument('--alg', type=str, default='fedph', help='algorithms: local, fedph, fedproto, fedavg, fedprox')
    parser.add_argument('--train_ep', type=int, default=1, help='the number of local episodes')
    parser.add_argument('--local_bs', type=int, default=32, help='local batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Adam weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer:SGD,Adam')

    parser.add_argument('--num_bb', type=int, default=1, help='number of backbone')

    parser.add_argument('--train_size', type=float, default=0.9, help='proportion of training dataset')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--alpha', type=float, default=1, help='parameters of probability distribution')
    parser.add_argument('--non_iid', type=int, default=2,
                        help='0:feature shift,1:label shift,2:feature shift and label shift')

    parser.add_argument('--ld', type=int, default=1, help='hyperparameter of fedproto and fedph')
    parser.add_argument('--mu', type=int, default=1, help='hyperparameter of fedprox')
    parser.add_argument('--temperature', type=int, default=1, help='hyperparameter of fedph')

    parser.add_argument('--is_not_the', type=int, default=0,
                        help='multi-key encryption scheme:0 is not enabled and 1 is enabled')

    parser.add_argument('--add_noise_proto', type=int, default=0,
                        help='differential privacy:0 is not enabled and 1 is enabled')
    parser.add_argument('--scale', type=float, default=0.01, help='noise distribution std')
    parser.add_argument('--noise_type', type=str, default="gaussian", help='noise type')
    args = parser.parse_args()
    return args


def args_parser():
    args = get_args()

    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.seed = 420
    args.classes = ['Bus',
                    'Microbus',
                    'Minivan',
                    'Sedan',
                    'SUV',
                    'Truck']
    args.domains = ['Sunny',
                    'Rainy',
                    'Snowy',
                    'Fog',
                    'Cloudy']
    args.file_path = './data/VehicleDataset-DallE'

    args.ways = 4
    args.shots = 500
    args.train_shots_max = 100
    args.test_shots = args.train_shots_max * (1 - args.train_size)
    args.stdev = 2
    args.train_num_shards = 0
    args.train_num_imgs = 10
    args.test_num_shards = 0
    args.test_num_imgs = 10

    args.threshold = args.num_users // 2 + 1
    return args