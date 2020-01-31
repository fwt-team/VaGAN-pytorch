# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: studentT_model_v2.py
@Time: 2019/10/14 下午4:45
@Desc: studentT_model_v2.py
"""

try:
    import os
    import math

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from scipy.special import digamma, gammaln
    from torch.distributions import Gamma, Chi2
    from sklearn.cluster import KMeans

    from vagan.utils import init_weights
    from vagan.config import DEVICE

except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )


class Generator(nn.Module):

    def __init__(self, latent_dim=50, x_shape=(1, 28, 28), cshape=(128, 7, 7), verbose=False):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ishape = cshape
        self.iels = int(np.prod(self.ishape))
        self.x_shape = x_shape
        self.output_channels = x_shape[0]
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(True),

            nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.ReLU(True),
            #
            Reshape(self.ishape),
        )

        # block_layers = []
        # for i in range(6):
        #     block_layers += [ResNetBlock(128)]

        # self.model = nn.Sequential(self.model, *block_layers)

        self.model = nn.Sequential(
            self.model,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(64, self.output_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):

        gen_img = self.model(x)
        # print(gen_img.size())
        return gen_img.view(x.size(0), *self.x_shape)


class Encoder(nn.Module):

    def __init__(self, input_channels=1, output_channels=64, cshape=(128, 7, 7), r=9, verbose=False):
        super(Encoder, self).__init__()

        self.cshape = cshape
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.r = r
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )

        self._mu = nn.Linear(1024, self.output_channels)
        self._con = nn.Linear(1024, self.output_channels)
        self._v = nn.Linear(1024, 1)

        init_weights(self)
        self._v.weight.data.uniform_(0.01, 0.03)
        if self.verbose:
            print(self.model)

    def forward(self, x):
        # x.shape is [64, 1, 28, 28]
        z = self.model(x)
        mu = self._mu(z)
        log_sigma = self._con(z)
        log_v = self._v(z)
        v = torch.exp(log_v) + self.r

        sigma = torch.exp(log_sigma * 0.5)

        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(DEVICE)

        # e = torch.tensor(np.random.normal(0, 1, sigma.size()), dtype=torch.float).to(DEVICE)
        e = torch.tensor(np.random.uniform(-1, 1, sigma.size()), dtype=torch.float).to(DEVICE)

        # sample from Gamma distribution
        z1 = (v / 2 - 1 / 3) * torch.pow(1 + (e / torch.sqrt(9 * v / 2 - 3)), 3)

        # reparameterization trick
        z2 = torch.sqrt(v / (2 * z1))
        z = mu + sigma * z2 * std_z

        return z, mu, log_sigma - torch.log(z1)


class SMM(nn.Module):

    def __init__(self, n_cluster=10, n_features=64):
        super(SMM, self).__init__()

        self.n_cluster = n_cluster
        self.n_features = n_features

        self.pi_ = nn.Parameter(torch.FloatTensor(self.n_cluster, ).fill_(1) / self.n_cluster, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(self.n_cluster, self.n_features).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.n_cluster,
                                                           self.n_features).fill_(0), requires_grad=True)

    def predict(self, z):

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))

        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def get_pro(self, z):
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))

        yita = yita_c.detach().cpu().numpy()
        return yita

    def sample(self, z):

        pred = self.predict(z)
        Z = None
        for i in pred:
            mu = self.mu_c[i].data.cpu().numpy()
            sigma = self.log_sigma2_c[i].exp().data.cpu().numpy()
            z_i = np.concatenate(mu + np.random.randn(1, self.n_features) * np.sqrt(sigma), 0)
            if Z is None:
                Z = z_i
            else:
                Z = np.vstack((Z, z_i))
        return torch.from_numpy(Z).to(DEVICE).float()

    def sample_by_k(self, k, num=10):

        mu = self.mu_c[k].data.cpu().numpy()
        sigma = self.log_sigma2_c[k].exp().data.cpu().numpy()
        z = mu + np.random.randn(num, self.n_features) * np.sqrt(sigma)
        return torch.from_numpy(z).to(DEVICE).float()

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):

        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):

        return -0.5 * (torch.sum(torch.tensor(np.log(np.pi * 2), dtype=torch.float).to(DEVICE) +
                                 log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1))

    def kl_Loss(self, z, z_mu, z_sigma2_log):

        det = 1e-10

        pi = self.pi_
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c

        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters

        Loss = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                             torch.exp(z_sigma2_log.unsqueeze(1) -
                                                                       log_sigma2_c.unsqueeze(0)) +
                                                             (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) /
                                                             torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * \
                torch.mean(torch.sum(1 + z_sigma2_log, 1))
        return Loss


class Discriminator(nn.Module):

    def __init__(self, input_channels=1, verbose=False, cshape=(128, 7, 7), w_distance=True):
        super(Discriminator, self).__init__()

        self.channels = input_channels
        self.cshape = cshape
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Linear(1024, 1),
        )

        if w_distance is False:
            self.model = nn.Sequential(
                self.model,
                nn.Sigmoid()
            )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, img):

        return self.model(img)
