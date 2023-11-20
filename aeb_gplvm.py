import warnings
import json
import random

warnings.filterwarnings("ignore")
# import the necessary package
from baseline.OE_GPLVM.aeb_gplvm import AEB_GPLVM, NNEncoder, kl_gaussian_loss_term
from baseline.OE_GPLVM.train import *
from baseline.OE_GPLVM.utils import *
from baseline.PyOD import PYOD
from gpytorch.mlls import KLGaussianAddedLossTerm
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, KLGaussianAddedLossTerm
from torch.distributions import kl_divergence
from gpytorch.priors import MultivariateNormalPrior
from tqdm import trange
from utils.data_generator import DataGenerator
from utils.myutils import Utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

plt.style.use("ggplot")
datagenerator = DataGenerator()  # data generator
utils = Utils()  # utils function

def get_indices(y_train):
    idx_a = np.where(y_train == 1)[0]
    idx_n = np.where(y_train == 0)[0]
    ratio = len(idx_a) / (len(idx_a) + len(idx_n))
    qtd_anomaly = int(ratio * batch_size)
    qtd_normal = batch_size - qtd_anomaly
    idx_n = torch.tensor(np.random.choice(idx_n, qtd_normal, replace=True))

    if qtd_anomaly == 0:
        idx_a = torch.tensor(np.random.choice(idx_n, qtd_anomaly, replace=True))
    else:
        idx_a = torch.tensor(np.random.choice(idx_a, qtd_anomaly, replace=True))

    batch_index = torch.cat([idx_n, idx_a])

    return idx_n, idx_a, batch_index, ratio


def get_loe_index(X, indices):
    ll_0, klu_0, kl_0, _ = calculate_terms(X, indices)
    score = ll_0 - kl_0

    qtd_normal = int(score.shape[0] * (1 - ratio))
    qtd_anormal = batch_size - int(score.shape[0] * (1 - ratio))

    _, loe_idx_n = torch.topk(score, qtd_normal, largest=True, sorted=False)
    _, loe_idx_a = torch.topk(score, qtd_anormal, largest=False, sorted=False)
    return indices[loe_idx_n], indices[loe_idx_a]


def create_dist_qx(model, batch_target):
    mu = model.predict_latent(batch_target)[0]
    sigma = model.predict_latent(batch_target)[1]
    local_q_x = MultivariateNormal(mu, sigma)
    return mu, sigma, local_q_x


def create_dist_prior(
    batch_target,
    mu,
):
    local_p_x_mean = torch.zeros(batch_target.shape[0], mu.shape[1])
    local_p_x_covar = torch.eye(mu.shape[1])
    local_p_x = MultivariateNormalPrior(local_p_x_mean, local_p_x_covar)
    return local_p_x


def kl_divergence_variational(target):
    ll_shape = torch.zeros_like(target.T)
    klu = (
        ll_shape.T.add_(model.variational_strategy.kl_divergence().div(batch_size))
        .sum(-1)
        .T.div((n_train))
    )
    return klu


def calculate_terms(X, indices):
    batch_target = X[indices]
    mu, sigma, local_q_x = create_dist_qx(model, batch_target)
    local_p_x = create_dist_prior(batch_target, mu)
    batch_output = model(model.sample_latent_variable(batch_target))
    log_likelihood = (
        likelihood.expected_log_prob(batch_target.T, batch_output)
        .sum(0)
        .div(batch_size)
    )
    kl_x = kl_divergence(local_q_x, local_p_x).div(n_train)
    kl_u = kl_divergence_variational(batch_target)
    log_marginal = (
        likelihood.log_marginal(batch_target.T, batch_output).sum(0).div(batch_size)
    )
    return log_likelihood, kl_u, kl_x, log_marginal


def predict_score(X_test):
    n_test = len(X_test)
    mu, sigma, local_q_x = create_dist_qx(model, X_test)
    local_p_x = create_dist_prior(X_test, mu)
    X_pred = model(model.sample_latent_variable(X_test))
    exp_log_prob = likelihood.expected_log_prob(X_test.T, X_pred)
    log_likelihood = exp_log_prob.sum(0).div(n_test)
    kl_x = kl_divergence(local_q_x, local_p_x).div(n_test)
    kl_u = kl_divergence_variational(X_test)
    score = -(log_likelihood - kl_u - kl_x).detach().numpy()
    score = MinMaxScaler().fit_transform(np.reshape(score, (-1, 1)))
    return score


records_dict = {
    "log_likelihood_normal": [],
    "kl_divergence_variational_normal": [],
    "kl_divergence_latent_normal": [],
    "log_marginal_normal": [],
    "log_likelihood_abnormal": [],
    "kl_divergence_variational_abnormal": [],
    "kl_divergence_latent_abnormal": [],
    "log_marginal_abnormal": [],
}


def save_records(records_dict, records):
    records_dict["log_likelihood_normal"].append(records[0].sum().detach().numpy())
    records_dict["kl_divergence_variational_normal"].append(
        records[1].sum().detach().numpy()
    )
    records_dict["kl_divergence_latent_normal"].append(
        records[2].sum().detach().numpy()
    )
    records_dict["log_marginal_normal"].append(records[3].sum().detach().numpy())
    records_dict["log_likelihood_abnormal"].append(records[4].sum().detach().numpy())
    records_dict["kl_divergence_variational_abnormal"].append(
        records[5].sum().detach().numpy()
    )
    records_dict["kl_divergence_latent_abnormal"].append(
        records[6].sum().detach().numpy()
    )
    records_dict["log_marginal_abnormal"].append(records[7].sum().detach().numpy())