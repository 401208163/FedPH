# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 17:38
# @Author  : Kuang Hangdong
# @File    : main.py
# @Software: PyCharm
# @desc    :
import copy

from tqdm import tqdm
import torch.utils.data

from tool.utils import *
from tool.model import *
from tool.privacy import *
from tool.options import *

from distro_paillier.distributed_paillier import NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PRIME_THRESHOLD, \
    STATISTICAL_SECURITY_SECRET_SHARING, CORRECTNESS_PARAMETER_BIPRIMALITY
from distro_paillier.distributed_paillier import generate_shared_paillier_key


def setup_seed(seed):
    """
    fixed random number seed
    :param seed:
    :return:
    """
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
        self.loss_mse = torch.nn.MSELoss().to(self.args.device)

    def train_val_test(self, train_image_paths, train_labels, test_image_paths, test_labels):
        """
        Divide training dataset and test dataset
        :param train_image_paths:
        :param train_labels:
        :param test_image_paths:
        :param test_labels:
        :return:
        """
        train_dataloader = torch.utils.data.DataLoader(Dataset(train_image_paths, train_labels),
                                                       batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(Dataset(test_image_paths, test_labels),
                                                      batch_size=self.args.local_bs, drop_last=True)

        return train_dataloader, test_dataloader

    def update_weights(self, round, backbone_list, global_model, model):
        """
        training mode: FedAVG,FedProx
        :param round:
        :param backbone_list:
        :param global_model:
        :param model:
        :return:
        """

        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_acc = []

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

    def update_weights_lg(self, global_protos, global_avg_protos, backbone_list, model):
        """
        training mode: FedPH,FedProto,Local
        :param global_protos:
        :param global_avg_protos:
        :param backbone_list:
        :param model:
        :return:
        """

        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], '1': [], '2': []}
        epoch_acc = []
        distance_function = torch.nn.CosineSimilarity(dim=-1).to(self.args.device)

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
                        loss2 = self.loss_mse(features_global, features)
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
        """
        test mode: FedPH,FedProto,Local,FedAVG,FedProx
        :param backbone_list:
        :param model:
        :return:
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
    setup_seed(args.seed)

    # the threshold scheme
    if args.is_not_the:
        Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta = generate_shared_paillier_key(
            keyLength=128)

    train_image_paths, train_labels, test_image_paths, test_labels = split_data(args.classes, args.domains,
                                                                                args.train_size, args.file_path)
    if args.non_iid == 1:
        train_client_indexs = dirichlet_split_noniid(train_labels, args.alpha, args.num_users)
    elif args.non_iid == 2:
        args.train_num_shards = len(train_image_paths) // args.train_num_imgs
        args.test_num_shards = len(test_image_paths) // args.test_num_imgs

        train_size = args.train_num_shards * args.train_num_imgs
        test_size = args.test_num_shards * args.test_num_imgs
        train_image_paths, train_labels = train_image_paths[:train_size], train_labels[:train_size]
        test_image_paths, test_labels = test_image_paths[:test_size], test_labels[:test_size]

        n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1),
                                   args.num_users)
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users)

        train_dataset = Dataset(train_image_paths, train_labels)
        test_dataset = Dataset(test_image_paths, test_labels)
        train_client_indexs, classes_list = noniid(args.train_num_shards, args.train_num_imgs, args.num_users,
                                                   args.train_shots_max, args.num_classes, train_dataset, n_list,
                                                   k_list)
        test_client_indexs = noniid_lt(args.test_num_shards, args.test_num_imgs, args.num_users, test_dataset,
                                       classes_list)

    # model
    backbone_list = if_not_backbone(args.device, args.num_bb)
    local_model_list = []
    local_model = Model(1000 * args.num_bb, 128, args.num_classes).to(args.device)
    for _ in range(args.num_users):
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(copy.deepcopy(local_model))

    if args.alg == 'fedph' or args.alg == 'fedproto' or args.alg == 'local':
        global_protos = {}
        global_avg_protos = {}
        local_protos = {}
        train_acc_list, train_acc_std_list = [], []
        train_loss_list, train_loss_std_list, train_loss1_list, train_loss2_list = [], [], [], []
        test_acc_list, test_acc_std_list = [], []
        test_loss_list, test_loss_std_list = [], []
    if args.alg == 'fedph' or args.alg == 'fedproto' or args.alg == 'local':
        for round in tqdm(range(args.rounds)):
            print(f'\n | Global Training Round : {round} |\n')
            local_weights, local_loss1, local_loss2, local_loss_total, local_acc = [], [], [], [], []
            for idx in range(args.num_users):
                if args.non_iid == 1:
                    local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[train_client_indexs[idx]],
                                              train_labels=train_labels[train_client_indexs[idx]],
                                              test_image_paths=test_image_paths, test_labels=test_labels)
                elif args.non_iid == 2:
                    local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[
                        train_client_indexs[idx].astype(np.int64)],
                                              train_labels=train_labels[train_client_indexs[idx].astype(np.int64)],
                                              test_image_paths=test_image_paths[
                                                  test_client_indexs[idx].astype(np.int64)],
                                              test_labels=test_labels[test_client_indexs[idx].astype(np.int64)])
                weights, loss, protos, acc = local_model.update_weights_lg(global_protos=global_protos,
                                                                           global_avg_protos=global_avg_protos,
                                                                           backbone_list=backbone_list,
                                                                           model=copy.deepcopy(local_model_list[idx]))
                agg_protos = agg_func(protos)
                if args.add_noise_proto:
                    agg_protos = add_noise_proto(args.device, agg_protos, args.scale, args.noise_type, args.threshold,
                                                 args.is_not_the)
                local_weights.append(copy.deepcopy(weights))
                local_loss1.append(copy.deepcopy(loss['1']))
                local_loss2.append(copy.deepcopy(loss['2']))
                local_loss_total.append(copy.deepcopy(loss['total']))
                local_protos[idx] = copy.deepcopy(agg_protos)
                local_acc.append(copy.deepcopy(acc))

            for idx in range(args.num_users):
                local_model_list[idx].load_state_dict(local_weights[idx])

            if args.is_not_the:
                local_protos_ = {}
                for idx in range(args.num_users):
                    local_protos_[idx] = THE_Encryption(local_protos[idx], PublicKey)
                # update global protos
                global_avg_protos = proto_aggregation(local_protos_, args.is_not_the)
            else:
                global_avg_protos = proto_aggregation(local_protos, args.is_not_the)

            if args.is_not_the:
                global_avg_protos = THE_Decryption(args.device, global_avg_protos, Key, NUMBER_PLAYERS,
                                                   CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta)
            global_protos = copy.deepcopy(local_protos)

            acc_avg = np.mean(local_acc)
            acc_std = np.std(local_acc)
            loss1_avg = np.mean(local_loss1)
            loss2_avg = np.mean(local_loss2)
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
                if args.non_iid == 1:
                    local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[train_client_indexs[idx]],
                                              train_labels=train_labels[train_client_indexs[idx]],
                                              test_image_paths=test_image_paths, test_labels=test_labels)
                elif args.non_iid == 2:
                    local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[
                        train_client_indexs[idx].astype(np.int64)],
                                              train_labels=train_labels[train_client_indexs[idx].astype(np.int64)],
                                              test_image_paths=test_image_paths[
                                                  test_client_indexs[idx].astype(np.int64)],
                                              test_labels=test_labels[test_client_indexs[idx].astype(np.int64)])
                acc, loss = local_model.test_inference(backbone_list=backbone_list,
                                                       model=copy.deepcopy(local_model_list[idx]))
                local_acc.append(copy.deepcopy(acc))
                local_loss.append(copy.deepcopy(loss))
            acc_avg = np.mean(local_acc)
            acc_std = np.std(local_acc)
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
            print(f'\n | Global Training Round : {round} |\n')
            local_weights, local_loss, local_acc = [], [], []
            for idx in range(args.num_users):
                if args.non_iid == 1:
                    local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[train_client_indexs[idx]],
                                              train_labels=train_labels[train_client_indexs[idx]],
                                              test_image_paths=test_image_paths, test_labels=test_labels)
                elif args.non_iid == 2:
                    local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[
                        train_client_indexs[idx].astype(np.int64)],
                                              train_labels=train_labels[train_client_indexs[idx].astype(np.int64)],
                                              test_image_paths=test_image_paths[
                                                  test_client_indexs[idx].astype(np.int64)],
                                              test_labels=test_labels[test_client_indexs[idx].astype(np.int64)])

                weights, acc, loss = local_model.update_weights(round=round, backbone_list=backbone_list,
                                                                global_model=model,
                                                                model=copy.deepcopy(local_model_list[idx]))
                local_weights.append(copy.deepcopy(weights))
                local_acc.append(copy.deepcopy(acc))
                local_loss.append(copy.deepcopy(loss))

            for idx in range(args.num_users):
                local_model_list[idx].load_state_dict(local_weights[idx])

            # updating the global weights
            weights_avg = copy.deepcopy(local_weights[0])
            for k in weights_avg.keys():
                for i in range(1, len(local_weights)):
                    weights_avg[k] += local_weights[i][k]

                weights_avg[k] = torch.div(weights_avg[k], len(local_weights))

            global_weights = weights_avg

            # move the updated weights to our model state dict
            model.load_state_dict(global_weights)

            acc_avg = np.mean(local_acc)
            acc_std = np.std(local_acc)
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
                if args.non_iid == 1:
                    local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[train_client_indexs[idx]],
                                              train_labels=train_labels[train_client_indexs[idx]],
                                              test_image_paths=test_image_paths, test_labels=test_labels)
                elif args.non_iid == 2:
                    local_model = LocalUpdate(args=args, train_image_paths=train_image_paths[
                        train_client_indexs[idx].astype(np.int64)],
                                              train_labels=train_labels[train_client_indexs[idx].astype(np.int64)],
                                              test_image_paths=test_image_paths[
                                                  test_client_indexs[idx].astype(np.int64)],
                                              test_labels=test_labels[test_client_indexs[idx].astype(np.int64)])
                acc, loss = local_model.test_inference(backbone_list=backbone_list,
                                                       model=copy.deepcopy(local_model_list[idx]))
                local_acc.append(copy.deepcopy(acc))
                local_loss.append(copy.deepcopy(loss))

            acc_avg = np.mean(local_acc)
            acc_std = np.std(local_acc)
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

    print("-" * 100)

    print(f"train_acc:{list(np.round(np.array(train_acc_list), 3))}")
    print(f"train_acc_std:{list(np.round(np.array(train_acc_std_list), 3))}")
    print(f"train_loss:{list(np.round(np.array(train_loss_list), 3))}")
    print(f"train_loss_std:{list(np.round(np.array(train_loss_std_list), 3))}")

    print("-" * 100)

    print(f"test_acc:{list(np.round(np.array(test_acc_list), 3))}")
    print(f"test_acc_std:{list(np.round(np.array(test_acc_std_list), 3))}")
    print(f"test_loss:{list(np.round(np.array(test_loss_list), 3))}")
    print(f"test_loss_std:{list(np.round(np.array(test_loss_std_list), 3))}")
