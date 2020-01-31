# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: test.py
@Time: 2020-01-31 14:46
@Desc: test for VaGAN-GMM
"""
try:
    import os
    import argparse

    import torch
    import torch.nn as nn
    from tqdm import tqdm

    from vagan.datasets import dataset_list, get_dataloader
    from vagan.config import RUNS_DIR, DATASETS_DIR, DEVICE, DATA_PARAMS
    from vagan.model import Generator, GMM, Encoder
    from vagan.utils import cluster_acc, gmm_Loss
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Testing Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="vagan", help="Name of training run")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-v", "--version_name", dest="version_name", default="GMM")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name, args.version_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    models_dir = os.path.join(run_dir, 'models')

    n_cluster = 10

    data_params = DATA_PARAMS[dataset_name]
    pretrain_batch_size, train_batch_size, latent_dim, picture_size, cshape, data_size, \
    pre_epoch, pre_lr, train_lr = data_params

    # test detail var
    test_batch_size = 10000

    # net
    gen = Generator(latent_dim=latent_dim, x_shape=picture_size, cshape=cshape)
    gmm = GMM(n_cluster=n_cluster, n_features=latent_dim)
    encoder = Encoder(input_channels=picture_size[0], output_channels=latent_dim, cshape=cshape)

    # set device: cuda or cpu
    gen.to(DEVICE)
    encoder.to(DEVICE)
    gmm.to(DEVICE)

    gen.load_state_dict(torch.load(os.path.join(models_dir, "gen.pkl"), map_location=DEVICE))
    encoder.load_state_dict(torch.load(os.path.join(models_dir, "enc.pkl"), map_location=DEVICE))
    gmm.load_state_dict(torch.load(os.path.join(models_dir, "gmm.pkl"), map_location=DEVICE))

    test_dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                     batch_size=test_batch_size, train=False)

    epoch_bar = tqdm(range(0, 30))
    for epoch in epoch_bar:

        # =============================================================== #
        # ==============================test============================= #
        # =============================================================== #
        gen.eval()
        encoder.eval()
        gmm.eval()

        with torch.no_grad():
            _data, _target = next(iter(test_dataloader))
            _data, _target = _data.to(DEVICE), _target.numpy()

            _z, _, _ = encoder(_data)
            _pred = gmm.predict(_z)
            _acc = cluster_acc(_pred, _target)[0] * 100

            print("[VaGAN]: epoch: {}, acc: {}%".format(epoch,  _acc))


if __name__ == '__main__':
    main()
