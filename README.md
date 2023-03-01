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
| `model`                     | The model architecture. Options: `simple-cnn`, `resnet50` .|
| `alg` | The training algorithm. Options: `moon`, `fedavg`, `fedprox`, `local_training` |
| `dataset`      | Dataset to use. Options: `cifar10`. `cifar100`, `tinyimagenet`|
| `lr` | Learning rate. |
| `batch-size` | Batch size. |
| `epochs` | Number of local epochs. |
| `n_parties` | Number of parties. |
| `sample_fraction` | the fraction of parties to be sampled in each round. |
| `comm_round`    | Number of communication rounds. |
| `partition` | The partition approach. Options: `noniid`, `iid`. |
| `beta` | The concentration parameter of the Dirichlet distribution for non-IID partition. |
| `mu` | The parameter for MOON and FedProx. |
| `temperature` | The temperature parameter for MOON. |
| `out_dim` | The output dimension of the projection head. |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `device` | Specify the device to run the program. |
| `seed` | The initial seed. |


## Usage

Here is an example to run MOON on CIFAR-10 with a simple CNN:
```
python main.py --model=simple-cnn \
    --alg=FedPH \
    --lr=0.01 \
    --mu=5 \
    --epochs=10 \
    --comm_round=100 \
    --n_parties=10 \
    --partition=noniid \
    --beta=0.5 \
    --logdir='./logs/' \
    --datadir='./data/' \
```

## Tiny-VehicleDataset
You can download Tiny-VehicleDataset [here](https://www.kaggle.com/datasets/shamate2b/vehicledataset). 

## Hyperparameters
If you use the same setting as our papers, you can simply adopt the hyperparameters reported in our paper. If you try a setting different from our paper, please tune the hyperparameters of MOON. You may tune mu from \{0.001, 0.01, 0.1, 1, 5, 10\}. If you have sufficient computing resources, you may also tune temperature from \{0.1, 0.5, 1.0\} and the output dimension of projection head from \{64, 128, 256\}. 



## Citation

Please cite our paper if you find this code useful for your research.

