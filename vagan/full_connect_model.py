# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: model.py
@time: 2019/6/9 下午8:20
@desc: full-linear model
"""

try:
    import os
    import math

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from vagan.utils import init_weights
    from vagan.config import DEVICE

except ImportError as e:
    print(e)
    raise ImportError


def block(in_c, out_c):
    layers = [
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(True)
    ]
    return layers


class Generator(nn.Module):

    def __init__(self, latent_dim=50, input_dim=784, inter_dims=[500, 500, 2000], verbose=False):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.inter_dims = inter_dims
        self.verbose = verbose

        self.model = nn.Sequential(
            *block(self.latent_dim, self.inter_dims[-1]),
            *block(self.inter_dims[-1], self.inter_dims[-2]),
            *block(self.inter_dims[-2], self.inter_dims[-3]),
            nn.Linear(self.inter_dims[-3], self.input_dim),
            nn.Sigmoid()
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):

        gen = self.model(x)
        return gen


class Encoder(nn.Module):

    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], latent_dim=10, verbose=False):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.inter_dims = inter_dims
        self.verbose = verbose

        self.model = nn.Sequential(
            *block(self.input_dim, self.inter_dims[0]),
            *block(self.inter_dims[0], self.inter_dims[1]),
            *block(self.inter_dims[1], self.inter_dims[2]),
        )

        self.mu = nn.Linear(self.inter_dims[-1], self.latent_dim)
        self.con = nn.Linear(self.inter_dims[-1], self.latent_dim)

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        z = self.model(x)
        mu = self.mu(z)
        log_sigma = self.con(z)
        sigma = torch.exp(log_sigma * 0.5)
        std_z = torch.randn_like(mu)

        z = mu + (sigma * std_z)
        return z, mu, log_sigma


class Encoder_SMM(nn.Module):

    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], r=9, latent_dim=10, verbose=False):
        super(Encoder_SMM, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.inter_dims = inter_dims
        self.r = r
        self.verbose = verbose

        self.model = nn.Sequential(
            *block(self.input_dim, self.inter_dims[0]),
            *block(self.inter_dims[0], self.inter_dims[1]),
            *block(self.inter_dims[1], self.inter_dims[2]),
        )

        self.mu = nn.Linear(self.inter_dims[-1], self.latent_dim)
        self.con = nn.Linear(self.inter_dims[-1], self.latent_dim)
        self.v = nn.Linear(self.inter_dims[-1], 1)

        init_weights(self)
        self.v.weight.data.uniform_(0.01, 0.03)
        if self.verbose:
            print(self.model)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        z = self.model(x)
        mu = self.mu(z)
        log_sigma = self.con(z)
        log_v = self.v(z)
        v = torch.exp(log_v) + self.r

        sigma = torch.exp(log_sigma * 0.5)

        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(DEVICE)

        e = torch.tensor(np.random.uniform(-1, 1, sigma.size()), dtype=torch.float).to(DEVICE)

        # sample from Gamma distribution
        z1 = (v / 2 - 1 / 3) * torch.pow(1 + (e / torch.sqrt(9 * v / 2 - 3)), 3)

        # reparameterization trick
        z2 = torch.sqrt(v / (2 * z1))
        z = mu + sigma * z2 * std_z

        return z, mu, log_sigma - torch.log(z1)


class Discriminator(nn.Module):

    def __init__(self, verbose=False, input_dim=784, inter_dims=[500, 500, 2000]):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.inter_dims = inter_dims
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.inter_dims[0]),
            nn.LeakyReLU(0.02, True),
            nn.Linear(self.inter_dims[0], self.inter_dims[1]),
            nn.LeakyReLU(0.02, True),
            nn.Linear(self.inter_dims[1], self.inter_dims[2]),
            nn.LeakyReLU(0.02, True),

            nn.Linear(self.inter_dims[2], 1),
            nn.Sigmoid()
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        return self.model(x)
