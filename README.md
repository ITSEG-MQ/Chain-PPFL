# Chain-PPFL

This is partly the reproduction of the paper of [Privacy-Preserving Federated Learning Framework Based on Chained Secure Multi-party Computing]   
Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

The testing accuracy of MLP on MINST: 92.14% (10 epochs training) with the learning rate of 0.01.
The testing accuracy of CNN on MINST: 98.37% (10 epochs training) with the learning rate of 0.01.

## FedAVG

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 

## DP-based FL

Federated learning with DP is produced by:
> python [main_fed_dp.py](main_fed_dp.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed_dp.py --dataset mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 

Note: Our paper leverges the Laplace mechanism to realise ϵ-Differential Privacy.

## FL based on Chain-PPFL

Federated learning with DP is produced by:
> python [main_fed_serial.py](main_fed_serial.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed_serial.py --dataset mnist --num_channels 1 --model cnn --epochs 50 --gpu 0 

## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model  | Acc. of IID | Acc. of Non-IID|
| -----  | -----       | ----           |
| MLP    |  83.85%     | 74.00%         |
| CNN    |  96.27%     | 78.79%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model  | Acc. of IID | Acc. of Non-IID|
| -----  | -----       | ----           |
| MLP    | 93.49%      | 87.83%         |
| CNN    | 98.54%      | 96.30%         |

## References
```
@article{mcmahan2016communication,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, H Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and others},
  journal={arXiv preprint arXiv:1602.05629},
  year={2016}
}

@article{ji2018learning,
  title={Learning Private Neural Language Modeling with Attentive Aggregation},
  author={Ji, Shaoxiong and Pan, Shirui and Long, Guodong and Li, Xue and Jiang, Jing and Huang, Zi},
  journal={arXiv preprint arXiv:1812.07108},
  year={2018}
}

@article{Li2020privacy,
  title={Privacy-Preserving Federated Learning Framework Based on Chained Secure Multi-party Computing},
  author={Li, Yong and Zhou, Yipeng and Jolfaei, Alireza and Yu, Dongjin and Xu, Gaochao and Zheng, Xi},
  journal={IEEE Internet of Things Journal},
  year={2020}
}
```

Chain-PPFL [[Code](https://github.com/ITSEG-MQ/Chain-PPFL)]

## Requirements
python 3.6  
pytorch>=0.4
