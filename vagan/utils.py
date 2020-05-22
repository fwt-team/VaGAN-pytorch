# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: utils.py
@time: 2019/6/9 下午8:20
@desc: utils function
"""

try:
    import os
    import torch
    import argparse
    import numpy as np
    import torch.nn.functional as F
    import pandas as pd
    import torch.nn as nn

    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    from torchvision.utils import save_image
    from vagan.config import DEVICE, DATASETS_DIR, THRESHOLD
    from scipy.optimize import linear_sum_assignment as linear_assignment

except ImportError as e:
    print(e)
    raise ImportError


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class Noisy_Data:

    def __init__(self):

        self.datas = None

    def get_datas(self, data_name):

        if self.datas is None:
            path = os.path.join(DATASETS_DIR, '{}_noisy.csv'.format(data_name))
            self.datas = pd.read_csv(path, index_col=[0]).values
            return self.datas
        else:
            return self.datas


noisy_data = Noisy_Data()
def get_one_data_from_noisy(data_name='mnist', size=(1, 28, 28)):

    if size[0] == 1:
        data_name = 'mnist'
    elif size[0] == 3:
        data_name = 'cifar10'

    datas = noisy_data.get_datas(data_name)

    # ensure the index is useful
    index = int(np.random.uniform(0, datas.shape[0]))
    for i in range(size[0]):
        if index % size[0] != 0:
            index -= 1
        else:
            break
    if size[0] > 1 and index == datas.shape[0]-1:
        index = index - size[0] + 1

    data = torch.tensor(datas[index: index+size[0]], dtype=torch.float).view(*size)
    return data / 255


def init_weights(net):

    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def save_images(gen_imgs, imags_path, images_name, nrow=5):

    save_image(gen_imgs.data[:nrow * nrow],
               '%s/%s.png' % (imags_path, images_name),
               nrow=nrow, normalize=True)


def calc_gradient_penalty(netD, real_data, generated_data, x_size=1):

    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    if x_size == 1:
        # CNN: size[b_size, input_channels, width, height]
        alpha = torch.rand(b_size, 1, 1, 1)
    else:
        # DNN linear: size[b_size, input_dim]
        real_data = real_data.view(b_size, -1)
        alpha = torch.rand(b_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(DEVICE)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(DEVICE)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(DEVICE),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


def cluster_acc(Y_pred, Y):

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total*1.0/Y_pred.size, w


def gmm_Loss(z_mu, z_sigma2_log, gmm):

    # if isinstance(gmm, nn.DataParallel):
    #     gmm = gmm.module

    det = 1e-10

    z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu

    pi = gmm.pi_
    mu_c = gmm.mu_c
    log_sigma2_c = gmm.log_sigma2_c

    yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + gmm.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

    yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters

    Loss = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                          torch.exp(z_sigma2_log.unsqueeze(1) -
                                                                    log_sigma2_c.unsqueeze(0)) +
                                                          (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) /
                                                          torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

    Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * \
            torch.mean(torch.sum(1 + z_sigma2_log, 1))
    return Loss
