from dataclasses import dataclass, asdict, field
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.mlls.added_loss_term import AddedLossTerm
from gpytorch.models import ApproximateGP
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from prettytable import PrettyTable
from torch import nn
from torch.distributions import kl_divergence
from typing import List
import gpytorch
import numpy as np
import os
import pickle as pkl
import sys
import torch
import torch.nn.functional as F
from tqdm import trange


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


class NN_Encoder(LatentVariable):
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

        q_x = torch.distributions.MultivariateNormal(mu, sg)

        prior_x = self.prior_x
        prior_x.loc = prior_x.loc[: len(Y), ...]
        prior_x.covariance_matrix = prior_x.covariance_matrix[: len(Y), ...]

        x_kl = kl_gaussian_loss_term(q_x, prior_x, self.n, self.data_dim)
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


class GP_Decoder(BayesianGPLVM):
    def __init__(
        self, n, data_dim, latent_dim, n_inducing, X, nn_layers=None, kernel=None
    ):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        self.inducing_inputs = torch.randn(n_inducing, latent_dim)

        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(
            self, self.inducing_inputs, q_u, learn_inducing_locations=True
        )
        super(GP_Decoder, self).__init__(X, q_f)

        # Kernel
        # self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.mean_module = ZeroMean()
        if not kernel:
            print("Setando Kernel RBF Padrão")
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        else:
            if kernel == "rbf":
                self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
            elif kernel == "matern_5_2":
                self.covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=latent_dim)
                )
            elif kernel == "matern_3_2":
                self.covar_module = ScaleKernel(
                    MaternKernel(nu=1.5, ard_num_dims=latent_dim)
                )
            else:
                raise NotImplemented

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


class AD_GPLVM:
    def __init__(
        self,
        latent_dim: int,
        n_inducing: int,
        n_epochs: int,
        nn_layers: tuple,
        lr: float,
        batch_size: int,
    ) -> None:
        self.latent_dim = latent_dim
        self.n_inducing = n_inducing
        self.n_epochs = n_epochs
        self.nn_layers = nn_layers
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, Y_train):
        n_train = len(Y_train)
        data_dim = Y_train.shape[1]

        X_prior_mean = torch.zeros(n_train, self.latent_dim)
        X_prior_covar = torch.eye(X_prior_mean.shape[1])
        prior_x = MultivariateNormalPrior(X_prior_mean, X_prior_covar)

        encoder = NN_Encoder(
            n_train, self.latent_dim, prior_x, data_dim, self.nn_layers
        )

        model = GP_Decoder(
            n_train,
            data_dim,
            self.latent_dim,
            self.n_inducing,
            encoder,
            self.nn_layers,
        )

        likelihood = GaussianLikelihood()

        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            self.lr,
        )

        loss_list = []

        elbo = VariationalELBO(likelihood, model, num_data=len(Y_train))
        model.train()

        iterator = trange(1, leave=True)

        for i in iterator:
            batch_index = model._get_batch_idx(self.batch_size)
            optimizer.zero_grad()
            sample = model.sample_latent_variable(Y_train)
            sample_batch = sample[batch_index]
            output_batch = model(sample_batch)
            print(sample.shape)
            print(output_batch)
            loss = -elbo(output_batch, Y_train[batch_index].T).sum()
            loss_list.append(loss.item())
            iterator.set_description(
                "Loss: " + str(float(np.round(loss.item(), 2))) + ", iter no: " + str(i)
            )
            loss.backward()
            optimizer.step()

    def predict_score(self, X_test):
        with torch.no_grad():
            self.model.eval()
            self.likelihood.eval()
