# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: config.py
@time: 2019/8/9 下午1:03
@desc: config
"""

import os
import torch

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Local directory of CypherCat API
VAGAN_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
REPO_DIR = os.path.split(VAGAN_DIR)[0]

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')

# Local directory for runs
RUNS_DIR = os.path.join(REPO_DIR, 'runs')

THRESHOLD = 0.05

# difference datasets config
# pretrain_batch_size, train_batch_size, latent_dim, picture_size, cshape, all_data_size, pre_epoch, pre_lr, train_lr
DATA_PARAMS = {
    'mnist': (800, 64, 7, (1, 28, 28), (128, 7, 7), 60000, 50, 1e-3, 2e-3),
    'fashion-mnist': (250, 64, 9, (1, 28, 28), (128, 7, 7), 60000, 10, 1e-4, 1e-4),
    'cifar10': (128, 64, 12, (3, 32, 32), (128, 8, 8), 50000, 10, 1e-4, 1e-4),
    'reuters10k': (64, 64, 7, (), (128, 10, 49), 9000, 10, 1e-4, 1e-4),
}
DATA_PARAMS_DNN = {
    'mnist': (800, 64, 7, (1, 28, 28), (128, 7, 7), 60000, 50, 1e-3, 2e-3),
    'fashion-mnist': (128, 64, 7, (1, 28, 28), (128, 7, 7), 60000, 10, 1e-3, 1e-4),
    'cifar10': (128, 64, 12, (3, 32, 32), (128, 8, 8), 50000, 10, 1e-4, 1e-4),
    'reuters10k': (128, 64, 7, (), (), 9000, 50, 1e-4, 1e-4),
    'reuters': (100, 100, 7, (), (), 616500, 10, 1e-4, 1e-4),
}
DATA_PARAMS_SMM = {
    'mnist': (800, 64, 7, (1, 28, 28), (128, 7, 7), 60000, 40, 1e-3, 1e-3),
    'fashion-mnist': (128, 64, 9, (1, 28, 28), (128, 7, 7), 60000, 15, 1e-4, 1e-4),
    'cifar10': (128, 64, 12, (3, 32, 32), (128, 8, 8), 50000, 10, 1e-4, 1e-4),
    'reuters10k': (64, 64, 7, (), (), 9000, 10, 1e-4, 1e-4),
}
