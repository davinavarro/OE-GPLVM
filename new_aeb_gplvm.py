from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.mlls.added_loss_term import AddedLossTerm
from gpytorch.models import ApproximateGP
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
        self.kl_per_all = kl_divergence(self.q_x, self.p_x)
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
                # mu = mu * 5
                mu = mu * 1
        return mu

    def sigma(self, Y):
        sg = torch.tanh(self.sg_layers[0](Y))
        for i in range(1, len(self.sg_layers)):
            sg = torch.tanh(self.sg_layers[i](sg))
            if i == (len(self.sg_layers) - 1):
                # sg = sg * 5
                sg = sg * 1

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
        self.kl_latent = x_kl
        self.qxrsample = q_x.rsample()
        return self.qxrsample

    # def forward(self, Y, batch_idx=None, test=False):
    #    mu = self.mu(Y)
    #    sg = self.sigma(Y)


#
#    if batch_idx is None:
#        batch_idx = np.arange(self.n)
#
#    mu = mu[batch_idx, ...]
#    sg = sg[batch_idx, ...]
#
#    q_x = torch.distributions.MultivariateNormal(mu, sg)
#
#    prior_x = self.prior_x
#    prior_x.loc = prior_x.loc[: len(batch_idx), ...]
#    prior_x.covariance_matrix = prior_x.covariance_matrix[: len(batch_idx), ...]
#
#    x_kl = kl_gaussian_loss_term(q_x, self.prior_x, len(batch_idx), self.data_dim)
#    self.update_added_loss_term("x_kl", x_kl)
#    self.kl_latent = x_kl
#    self.qxrsample = q_x.rsample()
#    return q_x.rsample()


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
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        else:
            if kernel == "rbf":
                self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
            elif kernel == "matern_1_2":
                self.covar_module = ScaleKernel(
                    MaternKernel(nu=0.5, ard_num_dims=latent_dim)
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
        kernel: str = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.n_inducing = n_inducing
        self.n_epochs = n_epochs
        self.nn_layers = nn_layers
        self.lr = lr
        self.batch_size = batch_size
        self.kernel = kernel

    def fit(self, Y_train, lb_train=None, loss_type="blind", tune=False):
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        if lb_train is not None:
            lb_train = torch.tensor(lb_train, dtype=torch.float32)
            self.contamination = (lb_train.sum() / len(lb_train)).item()

        n_train = len(Y_train)
        self.n_train = n_train
        data_dim = Y_train.shape[1]

        X_prior_mean = torch.zeros(n_train, self.latent_dim)
        X_prior_covar = torch.eye(X_prior_mean.shape[1])
        prior_x = MultivariateNormalPrior(X_prior_mean, X_prior_covar)

        self.encoder = NN_Encoder(
            n_train, self.latent_dim, prior_x, data_dim, self.nn_layers
        )

        self.model = GP_Decoder(
            n_train,
            data_dim,
            self.latent_dim,
            self.n_inducing,
            self.encoder,
            self.nn_layers,
            self.kernel,
        )

        self.likelihood = GaussianLikelihood()

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            self.lr,
        )

        self.loss_list = []
        self.loe_list = []
        self.lll_elbo = []
        self.klx_elbo = []
        self.klu_elbo = []
        self.lll_loe = []
        self.klx_loe = []
        self.klu_loe = []

        self.elbo = VariationalELBO(
            self.likelihood, self.model, num_data=len(Y_train), combine_terms=True
        )
        self.model.train()

        for i in range(self.n_epochs):
            self.i = i
            batch_index = self.model._get_batch_idx(self.batch_size)
            self.batch_index = batch_index
            self.optimizer.zero_grad()
            sample = self.model.sample_latent_variable(Y_train)
            sample_batch = sample[batch_index]
            output_batch = self.model(sample_batch)

            loe_loss = self.calculate_loe_loss(
                Y_train, batch_index, output_batch, method=loss_type
            )

            if loe_loss > 2 * (self.loe_list[-1]) and self.i > 20:
                break

            loss = -self.elbo(output_batch, Y_train[batch_index].T).sum()
            self.loss_list.append(loss.item())

            # if i > 10 and loss_type in ["soft", "hard", "refine"]:
            #    loe_loss.backward()
            # else:
            #    loss.backward()

            if loss_type in ["soft", "hard", "refine"] and tune == "alt":
                # print("alt")
                if i % 2 == 0:
                    loe_loss.backward()
                else:
                    loss.backward()
            elif loss_type in ["soft", "hard", "refine"] and tune == "start":
                # print("start")
                if i > 10 == 0:
                    loe_loss.backward()
                else:
                    loss.backward()
            else:
                loe_loss.backward()

            self.optimizer.step()

    def calculate_loe_loss(self, Y_train, batch_index, output_batch, method="blind"):
        # Verosemelhança
        self.lll = (
            self.likelihood.expected_log_prob(Y_train[batch_index].T, output_batch)
            .sum(0)
            .div(len(batch_index))
            # .sum()
        )

        # Inducao
        inducing_loss = torch.zeros_like(self.lll)
        klu_total = self.klu = (
            self.model.variational_strategy.kl_divergence().div(self.n_train).sum()
        )
        inducing_loss.add_(klu_total)
        self.klu = inducing_loss / self.batch_size

        # Latente
        added_loss = torch.zeros_like(self.lll)
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(
                (added_loss_term.loss() * Y_train.shape[1]) / self.batch_size
            )
        self.klx = added_loss

        # ELBO
        self.pred = output_batch
        self.batch = Y_train[batch_index].T
        self.loss_n = -(self.lll - self.klu - self.klx)
        self.loss_a = -self.loss_n

        # loe_loss = -(self.lll - self.klu - self.klx).sum()

        self.lll_loe.append(-self.lll.sum().item())
        self.klu_loe.append(-self.klu.sum().item())
        self.klx_loe.append(-self.klx.sum().item())

        if method == "blind":
            loe_loss = self.loss_n
            self.loss_result = loe_loss

        elif method == "refine":
            _, idx_n = torch.topk(
                self.loss_n,
                int(self.loss_n.shape[0] * (1 - self.contamination)),
                largest=False,
                sorted=False,
            )

            loe_loss = self.loss_n[idx_n]
            self.loss_result = loe_loss

        elif method == "hard":
            _, idx_n = torch.topk(
                self.loss_n,
                int(self.loss_n.shape[0] * (1 - self.contamination)),
                largest=False,
                sorted=False,
            )
            _, idx_a = torch.topk(
                self.loss_n,
                int(self.loss_n.shape[0] * self.contamination),
                largest=True,
                sorted=False,
            )
            loe_loss = torch.cat(
                [
                    self.loss_n[idx_n],
                    self.loss_a[idx_a],
                ],
                0,
            )

            self.loss_result = loe_loss

        elif method == "soft":
            _, idx_n = torch.topk(
                self.loss_n,
                int(self.loss_n.shape[0] * (1 - self.contamination)),
                largest=False,
                sorted=False,
            )
            _, idx_a = torch.topk(
                self.loss_n,
                int(self.loss_n.shape[0] * self.contamination),
                largest=True,
                sorted=False,
            )

            loe_loss = torch.cat(
                [
                    self.loss_n[idx_n],
                    0.25 * self.loss_a[idx_a],
                ],
                0,
            )

            self.loss_result = loe_loss

        self.loe_list.append(loe_loss.sum().item())
        return loe_loss.sum()

    def calculate_train_elbo(self, Y_train):
        elbo_iter = 0
        for i in range(20):
            with torch.no_grad():
                self.model.eval()
                self.likelihood.eval()
                sample = self.model.sample_latent_variable(Y_train)
                output = self.model(sample)
                loss = -self.elbo(output, Y_train.T).sum()
            elbo_iter += float(loss)
        elbo_avg = elbo_iter / 20
        return elbo_avg

    def predict_score(self, X_test: torch.tensor):
        # sem scaling
        X_test = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            self.model.eval()
            self.likelihood.eval()

            n_test = len(X_test)

            mu = self.model.predict_latent(X_test)[0]
            sigma = self.model.predict_latent(X_test)[1]
            local_q_x = MultivariateNormal(mu, sigma)

            local_p_x_mean = torch.zeros(X_test.shape[0], mu.shape[1])
            local_p_x_covar = torch.eye(mu.shape[1])
            local_p_x = MultivariateNormalPrior(local_p_x_mean, local_p_x_covar)

            X_pred = self.model(self.model.sample_latent_variable(X_test))

            log_likelihood = (
                self.likelihood.expected_log_prob(X_test.T, X_pred).sum(0).div(n_test)
            )
            kl_x = kl_divergence(local_q_x, local_p_x).div(n_test)

            ll_shape = torch.zeros_like(X_test.T)
            klu = self.model.variational_strategy.kl_divergence().div(self.batch_size)
            klu_expanded = ll_shape.T.add_(klu).sum(-1).T.div((self.n_train))

            score = -(log_likelihood - klu_expanded - kl_x).detach().numpy()
            # score = MinMaxScaler().fit_transform(np.reshape(score, (-1, 1)))

            return score

    def get_2d_latent(self):
        inv_lengthscale = 1 / self.model.covar_module.base_kernel.lengthscale
        values, indices = torch.topk(
            self.model.covar_module.base_kernel.lengthscale, k=2, largest=False
        )

        l1 = indices.numpy().flatten()[0]
        l2 = indices.numpy().flatten()[1]

        X = self.model.X.q_mu.detach().numpy()
        std = torch.nn.functional.softplus(self.model.X.q_log_sigma).detach().numpy()
        return X, std
