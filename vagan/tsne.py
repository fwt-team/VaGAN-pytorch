# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: tsne.py
@time: 2019/6/9 下午8:20
@desc: tsne
"""

import os

import torch
import matplotlib.pyplot as plt
import numpy as np

from vagan.config import DATA_PARAMS, DATASETS_DIR, RUNS_DIR
from sklearn import manifold
from vagan.model import Encoder
from vagan.datasets import get_dataloader


def tsne(data_name='mnist'):

    run_dir = os.path.join(RUNS_DIR, data_name, "vagan", 'GMM')
    models_dir = os.path.join(run_dir, 'models')

    data_params = DATA_PARAMS[data_name]
    pretrain_batch_size, train_batch_size, latent_dim, picture_size, cshape, data_size, \
    pre_epoch, pre_lr, train_lr = data_params

    # net
    encoder = Encoder(input_channels=picture_size[0], output_channels=latent_dim, cshape=cshape)
    encoder.load_state_dict(torch.load(os.path.join(models_dir, "enc.pkl"), map_location=torch.device('cpu')))

    data_dir = os.path.join(DATASETS_DIR, data_name)
    dataloader = get_dataloader(dataset_path=data_dir, dataset_name=data_name,
                                batch_size=pretrain_batch_size, train=True)

    X = []
    Y = []
    for index, (data, y) in enumerate(dataloader):
        if index is 5:break
        Y.append(y.numpy())
        z, _, _ = encoder(data)
        X.append(z.detach().numpy())

    X = np.concatenate(X, 0)
    Y = np.concatenate(Y, 0)

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))

    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(Y[i]))
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./tsne.png', dpi=100)
    plt.show()

tsne()


