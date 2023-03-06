# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 17:38
# @Author  : Kuang Hangdong
# @File    : main.py
# @Software: PyCharm
# @desc    :
import torch
from tqdm import tqdm
import numpy as np
import random
import copy

from tool.utils import split_data, Dataset, criterion_CL, agg_func, proto_aggregation
from tool.model import if_not_backbone, Model


class args_parser():
    def __init__(self):
        # Federated Learning Arguments
        self.mode = 'syn'  # syn asy
        self.rounds = 50
        self.num_users = 5
        self.alg = 'fedph'  # local, fedph, fedproto, fedavg, fedprox
        self.train_ep = 1
        self.local_bs = 32
        self.lr = 0.001
        self.momentum = 0.5
        self.weight_decay = 1e-4
        self.optimizer = 'sgd'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.seed = 420

        # Model Arguments
        self.num_bb = 1
        self.backbone = 'resnet'
        self.model = 'mlp'

        # Data Arguments
        self.dataset = 'vehicles'
        self.train_size = 0.8
        self.num_classes = 6
        self.alpha = 1
        self.non_iid = 0

        # Loss Funtion
        self.distance = 'cos'
        self.ld = 1  # fedproto fedph
        self.mu = 1  # fedprox
        self.temperature = 0.5  # fedph


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.deterministic = True


class LocalUpdate(object):
    def __init__(self, args, train_image_paths, train_labels, test_image_paths, test_labels):
        self.args = args
        self.train_dataloader, self.test_dataloader = self.train_val_test(train_image_paths, train_labels,
                                                                          test_image_paths, test_labels)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train_val_test(self, train_image_paths, train_labels, test_image_paths, test_labels):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        train_dataloader = torch.utils.data.DataLoader(Dataset(train_image_paths, train_labels),
                                                       batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(Dataset(test_image_paths, test_labels),
                                                      batch_size=self.args.local_bs, drop_last=True)

        return train_dataloader, test_dataloader

    # def update_weights(self, idx, backbone_list, model, global_round):
    def update_weights(self, round, backbone_list, global_model, model):
        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_acc = []

        # if self.args.alg == 'fedprox':
        #     # use the weights of global model for proximal term calculation
        #     global_model = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        for iter in range(self.args.train_ep):
            batch_loss, batch_acc = [], []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # generate representations by different backbone
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)

                # compute loss
                model.zero_grad()
                log_probs, _ = model(reps)

                if self.args.alg == 'fedprox':
                    proximal_term = 0.0
                    if round != 0:
                        # iterate through the current and global model parameters
                        for w, w_t in zip(model.parameters(), global_model.parameters()):
                            # update the proximal term
                            proximal_term += (w - w_t).norm(2)
                    loss = self.criterion(log_probs, labels) + self.args.mu * proximal_term
                elif self.args.alg == 'fedavg':
                    loss = self.criterion(log_probs, labels)

                # optimizer
                loss.backward()
                optimizer.step()

                # acc
                log_probs = log_probs[:, 0:self.args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                batch_acc.append(acc_val.item())
                batch_loss.append(loss.item())
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_acc) / len(epoch_acc), sum(epoch_loss) / len(epoch_loss)

    # def update_weights_lg(self, args, idx, global_protos, global_avg_protos, backbone_list, model, global_round):
    def update_weights_lg(self, global_protos, global_avg_protos, backbone_list, model):
        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], '1': [], '2': []}
        epoch_acc = []
        loss_mse = torch.nn.MSELoss().to(self.args.device)
        if self.args.distance == 'cos':
            distance_function = torch.nn.CosineSimilarity(dim=-1).to(self.args.device)
        elif self.args.distance == 'l1':
            distance_function = torch.nn.PairwiseDistance(p=1).to(self.args.device)
        elif self.args.distance == 'l2':
            distance_function = torch.nn.PairwiseDistance(p=2).to(self.args.device)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        for iter in range(self.args.train_ep):
            batch_loss = {'1': [], '2': [], 'total': []}
            batch_acc = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                model.zero_grad()

                # generate representations by different backbone
                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)

                # compute supervised contrastive loss
                log_probs, features = model(reps)
                loss1 = self.criterion(log_probs, labels)

                # compute regularized loss term
                loss2 = 0 * loss1
                if len(global_protos) == self.args.num_users:
                    if self.args.alg == 'fedproto':
                        # compute global proto-based distance loss
                        num, xdim = features.shape
                        features_global = torch.zeros_like(features)
                        for i, label in enumerate(labels):
                            features_global[i, :] = copy.deepcopy(global_avg_protos[label.item()].data)
                        loss2 = loss_mse(features_global, features)
                    elif self.args.alg == 'fedph':
                        loss2 = criterion_CL(global_avg_protos, features, labels, distance_function, self.args)
                    elif self.args.alg == 'local':
                        pass

                loss = loss1 + self.args.ld * loss2
                # optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # acc
                log_probs = log_probs[:, 0:self.args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                batch_acc.append(acc_val.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        agg_protos_label = {}
        if self.args.alg != 'local':
            model.eval()
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)
                _, features = model(reps)
                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(features[i, :])
                    else:
                        agg_protos_label[labels[i].item()] = [features[i, :]]

        return model.state_dict(), epoch_loss, agg_protos_label, sum(epoch_acc) / len(epoch_acc)

    def test_inference(self, backbone_list, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        num_batches = len(self.test_dataloader)

        for batch_idx, (images, labels) in enumerate(self.test_dataloader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)

            # Inference
            log_probs, _ = model(reps)
            batch_loss = self.criterion(log_probs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss / num_batches

    def generate_protos(self, backbone_list, model):
        model.eval()
        agg_protos_label = {}
        for batch_idx, (images, labels) in enumerate(self.train_dataloader):
            images = images[0]
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return agg_protos_label


if __name__ == "__main__":
    args = args_parser()
    classes = [
        'Bus',
        'Microbus',
        'Minivan',
        'Sedan',
        'SUV',
        'Truck'
    ]
    domains = ['Sunny',
               'Rainy',
               'Snowy',
               'Fog',
               'Cloudy'
               ]
    file_path = './data/VehicleDataset-DallE'
    train_image_paths, train_labels, test_image_paths, test_labels, client_indexs = split_data(domains, classes,
                                                                                               file_path, args)
    # model
    backbone_list = if_not_backbone(args)
    local_model_list = []
    local_model = Model(512, 64, args.num_classes).to(args.device)
    for _ in range(args.num_users):
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    if args.alg == 'local' or args.alg == 'fedph' or args.alg == 'fedproto':
        global_protos = {}
        global_avg_protos = {}
        local_protos = {}
        train_acc_list, train_acc_std_list = [], []
        train_loss_list, train_loss_std_list, train_loss1_list, train_loss2_list = [], [], [], []
        test_acc_list, test_acc_std_list = [], []
        test_loss_list, test_loss_std_list = [], []
    if args.alg == 'local' or args.alg == 'fedph' or args.alg == 'fedproto':
        for round in tqdm(range(args.rounds)):
            # for round in range(args.rounds):
            print(f'\n | Global Training Round : {round} |\n')
            local_weights, local_loss1, local_loss2, local_loss_total, local_acc = [], [], [], [], []
            # idxs_users = np.arange(args.num_users)
            for idx in range(args.num_users):
                local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[client_indexs[idx]],
                                          train_labels=train_labels[client_indexs[idx]],
                                          test_image_paths=test_image_paths, test_labels=test_labels)
                w, loss, protos, acc = local_model.update_weights_lg(global_protos=global_protos,
                                                                     global_avg_protos=global_avg_protos,
                                                                     backbone_list=backbone_list,
                                                                     model=copy.deepcopy(local_model_list[idx]))
                agg_protos = agg_func(protos)

                local_weights.append(copy.deepcopy(w))
                local_loss1.append(copy.deepcopy(loss['1']))
                local_loss2.append(copy.deepcopy(loss['2']))
                local_loss_total.append(copy.deepcopy(loss['total']))
                local_protos[idx] = copy.deepcopy(agg_protos)
                local_acc.append(copy.deepcopy(acc))

            for idx in range(args.num_users):
                local_model_list[idx].load_state_dict(local_weights[idx])

            # update global protos
            global_avg_protos = proto_aggregation(local_protos)
            global_protos = copy.deepcopy(local_protos)

            # acc_avg = sum(local_acc) / len(local_acc)
            acc_avg = np.mean(local_acc)
            acc_std = np.std(local_acc)
            # loss1_avg = sum(local_loss1) / len(local_loss1)
            loss1_avg = np.mean(local_loss1)
            # loss2_avg = sum(local_loss2) / len(local_loss2)
            loss2_avg = np.mean(local_loss2)
            # loss_avg = sum(local_loss_total) / len(local_loss_total)
            loss_avg = np.mean(local_loss_total)
            loss_std = np.std(local_loss_total)

            print('Train Acc Mean: {:.3f} | Train Acc Std: {:.3f}'.format(acc_avg, acc_std))
            print('Train Loss Mean: {:.3f} | Train Loss Std: {:.3f}'.format(loss_avg, loss_std))
            train_acc_list.append(acc_avg)
            train_acc_std_list.append(acc_std)
            train_loss_list.append(loss_avg)
            train_loss_std_list.append(loss_std)
            train_loss1_list.append(loss1_avg)
            train_loss2_list.append(loss2_avg)

            local_acc, local_loss = [], []
            for idx in range(args.num_users):
                local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[client_indexs[idx]],
                                          train_labels=train_labels[client_indexs[idx]],
                                          test_image_paths=test_image_paths, test_labels=test_labels)
                acc, loss = local_model.test_inference(backbone_list=backbone_list,
                                                       model=copy.deepcopy(local_model_list[idx]))
                local_acc.append(copy.deepcopy(acc))
                local_loss.append(copy.deepcopy(loss))

            # acc_avg = sum(local_acc) / len(local_acc)
            acc_avg = np.mean(local_acc)
            acc_std = np.std(local_acc)
            # loss_avg = sum(local_loss) / len(local_loss)
            loss_avg = np.mean(local_loss)
            loss_std = np.std(local_loss)

            print('Test Acc Mean: {:.3f} | Test Acc Std: {:.3f}'.format(acc_avg, acc_std))
            print('Test Loss Mean: {:.3f} | Test Loss Std: {:.3f}'.format(loss_avg, loss_std))

            test_acc_list.append(acc_avg)
            test_acc_std_list.append(acc_std)
            test_loss_list.append(loss_avg)
            test_loss_std_list.append(loss_std)

    if args.alg == 'fedavg' or args.alg == 'fedprox':
        model = local_model

        # global model weights
        global_weights = model.state_dict()

        train_acc_list, train_acc_std_list = [], []
        train_loss_list, train_loss_std_list = [], []
        test_acc_list, test_acc_std_list = [], []
        test_loss_list, test_loss_std_list = [], []
    if args.alg == 'fedavg' or args.alg == 'fedprox':
        for round in tqdm(range(args.rounds)):
            # for round in range(args.rounds):
            w = []
            print(f'\n | Global Training Round : {round} |\n')
            local_weights, local_loss, local_acc = [], [], []
            for idx in range(args.num_users):
                local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[client_indexs[idx]],
                                          train_labels=train_labels[client_indexs[idx]],
                                          test_image_paths=test_image_paths, test_labels=test_labels)
                weights, acc, loss = local_model.update_weights(round=round, backbone_list=backbone_list,
                                                                global_model=model,
                                                                model=copy.deepcopy(local_model_list[idx]))

                w.append(copy.deepcopy(weights))
                local_acc.append(copy.deepcopy(acc))
                local_loss.append(copy.deepcopy(loss))

            print(list(local_model_list[0].parameters()))

            for idx in range(args.num_users):
                local_model_list[idx].load_state_dict(w[idx])

            print(list(local_model_list[0].parameters()))

            # updating the global weights
            weights_avg = copy.deepcopy(w[0])
            for k in weights_avg.keys():
                for i in range(1, len(w)):
                    weights_avg[k] += w[i][k]

                weights_avg[k] = torch.div(weights_avg[k], len(w))

            global_weights = weights_avg

            # move the updated weights to our model state dict
            model.load_state_dict(global_weights)

            # acc_avg = sum(local_acc) / len(local_acc)
            acc_avg = np.mean(local_acc)
            acc_std = np.std(local_acc)
            # loss_avg = sum(local_loss) / len(local_loss)
            loss_avg = np.mean(local_loss)
            loss_std = np.std(local_loss)

            print('Train Acc Mean: {:.3f} | Train Acc Std: {:.3f}'.format(acc_avg, acc_std))
            print('Train Loss Mean: {:.3f} | Train Loss Std: {:.3f}'.format(loss_avg, loss_std))
            train_acc_list.append(acc_avg)
            train_acc_std_list.append(acc_std)
            train_loss_list.append(loss_avg)
            train_loss_std_list.append(loss_std)

            local_acc, local_loss = [], []
            for idx in range(args.num_users):
                local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[client_indexs[idx]],
                                          train_labels=train_labels[client_indexs[idx]],
                                          test_image_paths=test_image_paths, test_labels=test_labels)
                acc, loss = local_model.test_inference(backbone_list=backbone_list,
                                                       model=copy.deepcopy(local_model_list[idx]))
                local_acc.append(copy.deepcopy(acc))
                local_loss.append(copy.deepcopy(loss))

            # acc_avg = sum(local_acc) / len(local_acc)
            acc_avg = np.mean(local_acc)
            acc_std = np.std(local_acc)
            # loss_avg = sum(local_loss) / len(local_loss)
            loss_avg = np.mean(local_loss)
            loss_std = np.std(local_loss)

            print('Test Acc Mean: {:.3f} | Test Acc Std: {:.3f}'.format(acc_avg, acc_std))
            print('Test Loss Mean: {:.3f} | Test Loss Std: {:.3f}'.format(loss_avg, loss_std))

            if args.alg == 'fedavg':
                for idx in range(args.num_users):
                    local_model_list[idx].load_state_dict(global_weights)

            test_acc_list.append(acc_avg)
            test_acc_std_list.append(acc_std)
            test_loss_list.append(loss_avg)
            test_loss_std_list.append(loss_std)
