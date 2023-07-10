import sys
import os

import pickle as pkl
import torch
import numpy as np
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpytorch.models import ApproximateGP
from prettytable import PrettyTable
import numpy as np
import torch

import gpytorch
import torch
from torch import nn
from torch.distributions import kl_divergence
from gpytorch.mlls.added_loss_term import AddedLossTerm
import torch.nn.functional as F
import numpy as np


class LatentVariable(gpytorch.Module):
    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.latent_dim = dim

    def forward(self, x):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class kl_gaussian_loss_term(AddedLossTerm):
    def __init__(self, q_x, p_x, n, data_dim):
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.data_dim = data_dim

    def loss(self):
        kl_per_latent_dim = kl_divergence(self.q_x, self.p_x).sum(axis=0)
        kl_per_point = kl_per_latent_dim.sum() / self.n  # scalar
        return kl_per_point / self.data_dim


class NNEncoder(LatentVariable):
    def __init__(self, n, latent_dim, prior_x, data_dim, layers):
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self._init_mu_nnet(layers)
        self._init_sg_nnet(len(layers))
        self.register_added_loss_term("x_kl")

    def _get_mu_layers(self, layers):
        return (self.data_dim,) + layers + (self.latent_dim,)

    def _init_mu_nnet(self, layers):
        layers = self._get_mu_layers(layers)
        n_layers = len(layers)

        self.mu_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(n_layers - 1)]
        )

    def _get_sg_layers(self, n_layers):
        n_sg_out = self.latent_dim**2
        n_sg_nodes = (self.data_dim + n_sg_out) // 2
        sg_layers = (self.data_dim,) + (n_sg_nodes,) * n_layers + (n_sg_out,)
        return sg_layers

    def _init_sg_nnet(self, n_layers):
        layers = self._get_sg_layers(n_layers)
        n_layers = len(layers)

        self.sg_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(n_layers - 1)]
        )

    def mu(self, Y):
        mu = torch.tanh(self.mu_layers[0](Y))
        for i in range(1, len(self.mu_layers)):
            mu = torch.tanh(self.mu_layers[i](mu))
            if i == (len(self.mu_layers) - 1):
                mu = mu * 5
        return mu

    def sigma(self, Y):
        sg = torch.tanh(self.sg_layers[0](Y))
        for i in range(1, len(self.sg_layers)):
            sg = torch.tanh(self.sg_layers[i](sg))
            if i == (len(self.sg_layers) - 1):
                sg = sg * 5

        sg = sg.reshape(len(sg), self.latent_dim, self.latent_dim)
        sg = torch.einsum("aij,akj->aik", sg, sg)

        jitter = torch.eye(self.latent_dim).unsqueeze(0) * 1e-5
        self.jitter = torch.cat([jitter for i in range(len(Y))], axis=0)

        return sg + self.jitter

    def forward(self, Y, batch_idx=None):
        mu = self.mu(Y)
        sg = self.sigma(Y)

        if batch_idx is None:
            batch_idx = np.arange(self.n)

        mu = mu[batch_idx, ...]
        sg = sg[batch_idx, ...]

        q_x = torch.distributions.MultivariateNormal(mu, sg)

        prior_x = self.prior_x
        prior_x.loc = prior_x.loc[: len(batch_idx), ...]
        prior_x.covariance_matrix = prior_x.covariance_matrix[: len(batch_idx), ...]

        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, len(batch_idx), self.data_dim)
        self.update_added_loss_term("x_kl", x_kl)
        return q_x.rsample()


class BayesianGPLVM(ApproximateGP):
    def __init__(self, X, variational_strategy):
        super(BayesianGPLVM, self).__init__(variational_strategy)
        self.X = X

    def forward(self):
        raise NotImplementedError

    def sample_latent_variable(self, *args, **kwargs):
        sample = self.X(*args, **kwargs)
        return sample

    def predict_latent(self, Y_test):
        mu_star = self.X.mu(Y_test)
        sigma_star = self.X.sigma(Y_test)
        return mu_star, sigma_star

    def get_X_mean(self, Y):
        return self.X.mu(Y).detach()

    def get_X_scales(self, Y):
        return np.array([torch.sqrt(x.diag()) for x in self.X.sigma(Y).detach()])

    def reconstruct_y(self, Y):
        y_pred = self(self.X.mu(Y))
        y_pred_mean = y_pred.loc.detach()
        y_pred_covar = y_pred.covariance_matrix.detach()
        return y_pred_mean, y_pred_covar

    def get_trainable_param_names(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")

    def store(self, losses, likelihood):
        self.losses = losses
        self.likelihood = likelihood


class AEB_GPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, X, nn_layers=None):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)

        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(
            self, self.inducing_inputs, q_u, learn_inducing_locations=True
        )
        super(AEB_GPLVM, self).__init__(X, q_f)

        # Kernel
        # self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

    def _get_individual_batch_idx(self, batch_size, y_train):
        size = y_train.shape[0]
        idx_a = np.where(y_train == 1)[0]
        idx_n = np.where(y_train == 0)[0]
        ratio = idx_a.shape[0] / size

        qtd_anomaly = int(ratio * batch_size)
        qtd_normal = batch_size - qtd_anomaly

        idx_n = torch.tensor(np.random.choice(idx_n, qtd_normal, replace=True))
        idx_a = torch.tensor(np.random.choice(idx_n, qtd_anomaly, replace=True))
        return torch.cat([idx_n, idx_a]) , ratio 

    def _get_normal_batch_idx(self, batch_size, y_train):
        idx_n = np.where(y_train == 0)[0]
        idx_n = np.random.choice(idx_n, batch_size, replace=True)

        return torch.tensor(idx_n)
