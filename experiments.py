import warnings
import random
import json

with open("experiments.json", "r") as file:
    experiments = json.load(file)

warnings.filterwarnings("ignore")

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
import uuid

plt.style.use("ggplot")
datagenerator = DataGenerator()
utils = Utils()


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


experiment_results = []
random.seed(42)
for experiment in experiments[:100]:
    dataset = experiment["dataset"]
    nn_layers = tuple(map(int, experiments[:1][0]["layers"].split(",")))
    kernel_type = experiment["kernel"]
    gplvm_type = experiment["loss"]

    if gplvm_type != "normal":
        continue

    labeled_anomalies = float(experiment["labeled_anomalies"])
    noise_type = experiment["noise_type"]
    n_epochs = int(experiment["n_epochs"])

    datagenerator.dataset = dataset
    data = datagenerator.generator(
        la=labeled_anomalies, realistic_synthetic_mode=None, noise_type=None
    )

    Y_train = torch.tensor(data["X_train"], dtype=torch.float32)
    Y_test = torch.tensor(data["X_test"], dtype=torch.float32)
    lb_train = torch.tensor(data["y_train"], dtype=torch.float32)
    lb_test = torch.tensor(data["y_test"], dtype=torch.float32)

    N = len(Y_train)

    print(dataset, nn_layers, N)
    data_dim = Y_train.shape[-1]
    latent_dim = int(experiment["latent_dim"])
    n_inducing = 50
    n_epochs = int(experiment["n_epochs"])
    lr = float(experiment["learning_rate"])
    batch_size = int(experiment["batch_size"])
    n_train = len(Y_train)

    model_dict = {}
    noise_trace_dict = {}
    loss_list = []
    noise_trace = []
    lln_list = []
    kln_list = []
    lla_list = []
    kla_list = []

    X_prior_mean = torch.zeros(n_train, latent_dim)
    X_prior_covar = torch.eye(X_prior_mean.shape[1])
    prior_x = MultivariateNormalPrior(X_prior_mean, X_prior_covar)
    encoder = NNEncoder(n_train, latent_dim, prior_x, data_dim, nn_layers)
    model = AEB_GPLVM(
        n_train,
        data_dim,
        latent_dim,
        n_inducing,
        encoder,
        nn_layers,
    )

    likelihood = GaussianLikelihood()
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr,
    )
    model.train()

    iterator = trange(n_epochs, leave=True)
    for i in iterator:
        optimizer.zero_grad()
        _, _, batch_index, ratio = get_indices(lb_train)
        idx_n, idx_a = get_loe_index(Y_train, batch_index)

        ll_n, klu_n, kl_n, lm_n = calculate_terms(Y_train, idx_n)
        ll_a, klu_a, kl_a, lm_a = calculate_terms(Y_train, idx_a)
        loss_normal, loss_anomaly = (ll_n - klu_n - kl_n).sum(), (
            ll_a - klu_a - kl_a
        ).sum()
        if gplvm_type == "normal":
            loss = -(loss_normal + loss_anomaly).sum()
        elif gplvm_type == "loe_hard":
            loss = -(loss_normal + loss_anomaly).sum()
        elif gplvm_type == "loe_soft":
            loss = -(loss_normal + loss_anomaly).sum()

        loss.backward()
        optimizer.step()
        iterator.set_description(
            "Loss: " + str(float(np.round(loss.item(), 2))) + ", iter no: "
        )
        if float(np.round(loss.item(), 2)) < -40:
            break

    with torch.no_grad():
        model.eval()
        likelihood.eval()
        Y_pred_mean, Y_pred_covar = model.reconstruct_y(Y_test)
        X_pred_mean, X_pred_covar = model.predict_latent(Y_test)
        metrics = utils.metric(y_true=lb_test, y_score=predict_score(Y_test))

    exp = {
        "gplvm_type": gplvm_type,
        "dataset": dataset,
        "noise_type": noise_type,
        "labeled_anomalies": labeled_anomalies,
        "data_dim": data_dim,
        "n_samples": N,
        "n_dim_latent": latent_dim,
        "n_epochs": n_epochs,
        "n_inducing": n_inducing,
        "n_layers": nn_layers,
        "learning_rate": lr,
        "batch_size": batch_size,
        "auc_roc": metrics["aucroc"],
        "auc_pr": metrics["aucpr"],
        "elbo_normal": [],
        "inv_lenghtscale": [],
    }

    experiment_results.append(exp)

    with open(f"results/experiment_{str(uuid.uuid4())}.json", "w") as final:
        json.dump(exp, final)
