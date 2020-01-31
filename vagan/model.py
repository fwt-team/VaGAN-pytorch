# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: model.py
@time: 2019/6/9 下午8:20
@desc: model
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

    def __init__(self, input_channels=1, output_channels=64, cshape=(128, 7, 7), verbose=False):
        super(Encoder, self).__init__()

        self.cshape = cshape
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.output_channels = output_channels
        self.input_channels = input_channels
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

        self.mu = nn.Linear(1024, self.output_channels)
        self.con = nn.Linear(1024, self.output_channels)

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        # x.shape is [64, 1, 28, 28]
        batch_size = x.size(0)
        z = self.model(x)
        mu = self.mu(z)
        log_sigma = self.con(z)
        sigma = torch.exp(log_sigma * 0.5)
        # std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(DEVICE)
        std_z = torch.randn_like(mu)

        z = mu + (sigma * std_z)
        return z, mu, log_sigma


class GMM(nn.Module):

    def __init__(self, n_cluster=10, r=1, n_features=64):
        super(GMM, self).__init__()

        self.n_cluster = n_cluster
        self.r = r
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
            G.append(self.gaussian_pdf_log(x, mus[c:c+1, :], log_sigma2s[c:c+1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):

        return -0.5*(torch.sum(torch.tensor(np.log(np.pi*2), dtype=torch.float).to(DEVICE) +
                               log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2), 1))


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

