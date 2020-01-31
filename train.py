# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: train.py
@time: 2019/6/9 下午8:18
@desc: training code of VaGAN-GMM
"""

try:
    import os
    import argparse
    from itertools import chain

    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm
    import torch.nn.functional as F

    from vagan.datasets import dataset_list, get_dataloader
    from vagan.config import RUNS_DIR, DATASETS_DIR, DEVICE, DATA_PARAMS
    from vagan.model import Generator, Discriminator, GMM, Encoder
    from vagan.utils import save_images, calc_gradient_penalty, init_weights, cluster_acc, gmm_Loss, str2bool
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="vagan", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=500, type=int, help="Number of epochs")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-v", "--version_name", dest="version_name", default="GMM")
    parser.add_argument("-ns", "--noisy", dest="noisy", default=False, type=str2bool)
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name, args.version_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')
    log_path = os.path.join(run_dir, 'logs')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # -----train-----
    # train detail var
    lr = 1e-4  # learning rate
    n_cluster = 10
    n_epochs = args.n_epochs
    n_skip_iter = 1
    b1 = 0.5
    b2 = 0.99  # 99
    decay = 2.5 * 1e-5
    w_distance = True

    data_params = DATA_PARAMS[dataset_name]
    pretrain_batch_size, train_batch_size, latent_dim, picture_size, cshape, data_size, \
    pre_epoch, pre_lr, train_lr = data_params

    if w_distance is not True:
        n_skip_iter = 4

    # gaussian var
    r = 1  # scaling factor for the covariance matrices.

    # test detail var
    test_batch_size = 10000

    # net
    gen = Generator(latent_dim=latent_dim, x_shape=picture_size, cshape=cshape)
    dis = Discriminator(input_channels=picture_size[0], w_distance=w_distance, cshape=cshape)
    gmm = GMM(n_cluster=n_cluster, r=r, n_features=latent_dim)
    encoder = Encoder(input_channels=picture_size[0], output_channels=latent_dim, cshape=cshape)

    xe_loss = nn.BCELoss(reduction="sum")
    mse_loss = nn.MSELoss()

    # parallel
    if torch.cuda.device_count() > 1:
        print("this GPU have {} core".format(torch.cuda.device_count()))
        nn.DataParallel(gen)
        nn.DataParallel(dis)
        nn.DataParallel(encoder)
        nn.DataParallel(gmm)

    # set device: cuda or cpu
    gen.to(DEVICE)
    dis.to(DEVICE)
    encoder.to(DEVICE)
    gmm.to(DEVICE)
    xe_loss.to(DEVICE)
    mse_loss.to(DEVICE)

    # optimization
    gen_enc_ops = torch.optim.Adam(chain(
        gen.parameters(),
        encoder.parameters(),
    ), lr=pre_lr, betas=(b1, b2), weight_decay=decay)
    gen_enc_gmm_ops = torch.optim.Adam(chain(
        gen.parameters(),
        encoder.parameters(),
        gmm.parameters(),
    ), lr=train_lr, betas=(b1, b2))

    lr_s = StepLR(gen_enc_gmm_ops, step_size=10, gamma=0.95)
    dis_ops = torch.optim.Adam(dis.parameters(), lr=lr, betas=(b1, b2))

    dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                batch_size=pretrain_batch_size, train=True)
    test_dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                     batch_size=test_batch_size, train=False)

    # =============================================================== #
    # ==========================pretraining========================== #
    # =============================================================== #
    pre_train_path = os.path.join(models_dir, 'pre_train')
    if not os.path.exists(pre_train_path):

        print('Pretraining......')
        epoch_bar = tqdm(range(pre_epoch))
        for _ in epoch_bar:
            L = 0
            for index, (x, y) in enumerate(dataloader):
                x = x.to(DEVICE)

                _, z, _ = encoder(x)
                x_ = gen(z)
                loss = xe_loss(x_, x) / pretrain_batch_size

                L += loss.detach().cpu().numpy()

                gen_enc_ops.zero_grad()
                loss.backward()
                gen_enc_ops.step()

            epoch_bar.write('Loss={:.4f}'.format(L / len(dataloader)))
            # save_images(x_, imgs_dir, 'pre_train_{}'.format(_), nrow=5)
        encoder.con.load_state_dict(encoder.mu.state_dict())

        _gmm = GaussianMixture(n_components=n_cluster, covariance_type='diag')
        Z = []
        Y = []
        with torch.no_grad():
            for index, (x, y) in enumerate(dataloader):
                x = x.to(DEVICE)

                _, z, _ = encoder(x)
                # assert F.mse_loss(z, z1) == 0
                Z.append(z)
                Y.append(y)

        Z = torch.cat(Z, 0).detach().cpu().numpy()
        Y = torch.cat(Y, 0).detach().numpy()

        pre = _gmm.fit_predict(Z)
        print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

        gmm.pi_.data = torch.from_numpy(_gmm.weights_).to(DEVICE).float()
        gmm.mu_c.data = torch.from_numpy(_gmm.means_).to(DEVICE).float()
        gmm.log_sigma2_c.data = torch.log(torch.from_numpy(_gmm.covariances_).to(DEVICE).float())

        os.makedirs(pre_train_path, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(pre_train_path, 'enc.pkl'))
        torch.save(gen.state_dict(), os.path.join(pre_train_path, 'gen.pkl'))
        torch.save(gmm.state_dict(), os.path.join(pre_train_path, 'gmm.pkl'))

    else:
        gen.load_state_dict(torch.load(os.path.join(pre_train_path, "gen.pkl"), map_location=DEVICE))
        encoder.load_state_dict(torch.load(os.path.join(pre_train_path, "enc.pkl"), map_location=DEVICE))
        gmm.load_state_dict(torch.load(os.path.join(pre_train_path, "gmm.pkl"), map_location=DEVICE))

    # =============================================================== #
    # ====================check the cheekpoint model================= #
    # =============================================================== #
    file_list = os.listdir(models_dir)
    max_dir_index = 0  # init max dir index val
    for file in file_list:
        # judge file is dir
        if os.path.isdir(os.path.join(models_dir, file)) and file.split("_")[0] == "cheekpoint":
            index = int(file.split("_")[1])
            if index > max_dir_index:
                max_dir_index = index

    if max_dir_index > 0:
        print("have cheekpoint file %d" % max_dir_index)
        # load cheekpoint file
        cheek_path = os.path.join(models_dir, "cheekpoint_{}".format(max_dir_index))
        gen.load_state_dict(torch.load(os.path.join(cheek_path, "gen.pkl")))
        dis.load_state_dict(torch.load(os.path.join(cheek_path, "dis.pkl")))
        encoder.load_state_dict(torch.load(os.path.join(cheek_path, "enc.pkl")))
        gmm.load_state_dict(torch.load(os.path.join(cheek_path, "gmm.pkl")))

    dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                batch_size=train_batch_size, train=True, noisy=args.noisy)
    # =============================================================== #
    # ============================training=========================== #
    # =============================================================== #
    epoch_bar = tqdm(range(max_dir_index, n_epochs))
    for epoch in epoch_bar:
        g_t_loss, d_t_loss = 0, 0
        for index, (real_images, target) in enumerate(dataloader):

            real_images, target = real_images.to(DEVICE), target.to(DEVICE)

            gen.train()
            gmm.train()
            encoder.train()
            encoder.zero_grad()
            gen.zero_grad()
            gmm.zero_grad()
            dis.zero_grad()
            gen_enc_gmm_ops.zero_grad()

            z, z_mu, z_sigma2_log = encoder(real_images)
            # z = gmm.sample(z)
            fake_images = gen(z)

            rec_loss = xe_loss(fake_images, real_images) * real_images.size(1) / train_batch_size

            D_real = dis(real_images)
            D_fake = dis(fake_images)

            if index % n_skip_iter == n_skip_iter - 1:

                # train generator, encoder and gmm
                if w_distance:
                    g_loss = torch.mean(D_fake) + gmm_Loss(z_mu, z_sigma2_log, gmm) + rec_loss
                else:
                    g_loss = torch.mean(-torch.log(D_fake)) + gmm_Loss(z_mu, z_sigma2_log, gmm) + rec_loss
                g_loss.backward(retain_graph=True)
                gen_enc_gmm_ops.step()
                g_t_loss += g_loss

            # train discriminator
            dis_ops.zero_grad()
            if w_distance:
                grad_penalty = calc_gradient_penalty(dis, real_images, fake_images)
                d_loss = torch.mean(D_real) - torch.mean(D_fake) + grad_penalty
            else:
                d_loss = torch.mean(-torch.log(D_real) - torch.log(1 - D_fake)) / 2
            d_loss.backward()
            dis_ops.step()
            d_t_loss += d_loss

        # save cheekpoint model
        if (epoch + 1) % 20 == 0:
            cheek_path = os.path.join(models_dir, "cheekpoint_{}".format(epoch))
            os.makedirs(cheek_path, exist_ok=True)
            torch.save(dis.state_dict(), os.path.join(cheek_path, 'dis.pkl'))
            torch.save(gen.state_dict(), os.path.join(cheek_path, 'gen.pkl'))
            torch.save(encoder.state_dict(), os.path.join(cheek_path, 'enc.pkl'))
            torch.save(gmm.state_dict(), os.path.join(cheek_path, 'gmm.pkl'))

        lr_s.step()
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

            stack_images = None
            for k in range(n_cluster):

                z = gmm.sample_by_k(k)
                fake_images = gen(z)

                if stack_images is None:
                    stack_images = fake_images[:n_cluster].data.cpu().numpy()
                else:
                    stack_images = np.vstack((stack_images, fake_images[:n_cluster].data.cpu().numpy()))
            stack_images = torch.from_numpy(stack_images)
            save_images(stack_images, imgs_dir, 'test_gen_{}'.format(epoch), nrow=n_cluster)

            logger = open(os.path.join(log_path, "log.txt"), 'a')
            logger.write(
                "[VaGAN]: epoch: {}, g_loss: {}, d_loss: {}, acc: {}%\n".format(epoch,
                                                                                  g_t_loss / len(dataloader),
                                                                                  d_t_loss / len(dataloader), _acc)
            )
            logger.close()
            print("[VaGAN]: epoch: {}, g_loss: {}, d_loss: {}, acc: {}%".format(epoch,
                                                                                  g_t_loss / len(dataloader),
                                                                                  d_t_loss / len(dataloader), _acc))

    torch.save(gen.state_dict(), os.path.join(models_dir, 'gen.pkl'))
    torch.save(dis.state_dict(), os.path.join(models_dir, 'dis.pkl'))
    torch.save(encoder.state_dict(), os.path.join(models_dir, 'enc.pkl'))
    torch.save(gmm.state_dict(), os.path.join(models_dir, 'gmm.pkl'))


if __name__ == '__main__':
    main()