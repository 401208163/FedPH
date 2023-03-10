# FedPH: Privacy-enhanced Heterogeneous Federated Learning
This is the code for paper [FedPH: Privacy-enhanced Heterogeneous Federated Learning](https://arxiv.org/abs/2301.11705).

**Abstract**: Federated Learning is a distributed machine-learning environment that allows clients to learn collaboratively without sharing private data. This is accomplished by exchanging parameters. However, the differences in data distributions and computing resources among clients make related studies difficult. To address these heterogeneous problems, we propose a novel Federated Learning method. Our method utilizes a pre-trained model as the backbone of the local model, with fully connected layers comprising the head. The backbone extracts features for the head, and the embedding vector of classes is shared between clients to improve the head and enhance the performance of the local model. By sharing the embedding vector of classes instead of gradient-based parameters, clients can better adapt to private data, and communication between the server and clients is more effective. To protect privacy, we propose a privacy-preserving hybrid method that adds noise to the embedding vector of classes. This method has a minimal effect on the performance of the local model when differential privacy is met. We conduct a comprehensive evaluation of our approach on a self-built vehicle dataset, comparing it with other Federated Learning methods under non-independent identically distributed(Non-IID).

## Dependencies
* PyTorch >= 1.0.0
* torchvision >= 0.2.1
* scikit-learn >= 0.23.1



## Parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `rounds`                     | number of rounds of training.|
| `num_users` | number of users.|
| `alg`      | algorithms. Options: `fedph`, `fedavg`, `fedprox`,` fedproto`, `local`.|
| `train_ep` | the number of local episodes.|
| `local_bs` | local batch size.|
| `lr` | learning rate.|
| `momentum` | SGD momentum.|
| `weight_decay` | Adam weight decay.|
| `optimizer`    | optimizer. Options: `SGD`, `Adam`.|
| `num_bb` | number of backbone.|
| `train_size` | proportion of training dataset.|
| `num_classes` | number of classes. |
| `alpha` | parameters of probability distribution.|
| `non_iid` | non-iid. Options:`0(feature shift)`,`1(label shift)`,`2(feature shift and label shift)`.|
| `ld` | hyperparameter of fedproto and fedph.|
| `mu` | hyperparameter of fedprox.|
| `is_not_the` | multi-key encryption scheme.`0 (is not enabled)`, `1 (is enabled)`.|
| `add_noise_proto` | differential privacy. `0 (is not enabled)`, `1 (is enabled)`.|
| `scale` | noise distribution std.|
| `noise_type` | noise type.|


## Usage

Here is an example to run FedPH on VehicleDataset:
```
python main.py --rounds=25 \
    --num_users=5 \
    --alg=fedph \
    --local_bs=32 \
    --train_size=0.9 \
    --non_iid=2 \
    --is_not_the=1 \
    --add_noise_proto=1 \
```

## Tiny-VehicleDataset
You can download VehicleDataset [here](https://www.kaggle.com/datasets/shamate2b/vehicledataset). 

## Hyperparameters
If you try a setting different from our paper, please tune the hyperparameters of FedPH. You may tune mu and ld from \{0.001, 0.01, 0.1, 1, 5, 10\}. If you have sufficient computing resources, you may also tune temperature from \{0.1, 0.5, 1.0\}. 



## Citation

Please cite our paper if you find this code useful for your research.
```latex
@article{hangdong2023fedhp,
  title={FedPH: Privacy-enhanced Heterogeneous Federated Learning},
  author={Hangdong, Kuang and Bo, Mi},
  journal={arXiv preprint arXiv:2301.11705},
  year={2023}
}
```
