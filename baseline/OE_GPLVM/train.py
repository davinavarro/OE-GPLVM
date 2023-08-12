import torch
from gpytorch.mlls import KLGaussianAddedLossTerm
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import MultivariateNormalPrior
import numpy as np
from torch.distributions import kl_divergence
from dataclasses import dataclass
from tqdm import trange
from baseline.OE_GPLVM.aeb_gplvm import AEB_GPLVM, NNEncoder
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.preprocessing import MinMaxScaler

BATCH_SIZE = 128
N_EPOCHS = 2000


@dataclass
class HyperParameters:
    latent_dim: int
    n_inducing: int
    nn_layers: tuple
    lr: float
    loe_alpha: float
    loe_beta: float
    kernel: str


# X_prior_mean = torch.zeros(experiment.N, latent_dim)
# X_prior_covar = torch.eye(X_prior_mean.shape[1])
# prior_x = MultivariateNormalPrior(X_prior_mean, X_prior_covar)
# encoder = NNEncoder(experiment.N, latent_dim, prior_x, data_dim, nn_layers)
# model = AEB_GPLVM(experiment.N, data_dim, latent_dim, n_inducing, encoder, nn_layers)
# likelihood = GaussianLikelihood()
# optimizer = torch.optim.Adam(
#    [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr
# )

"HARD_GPLVM"
"SOFT_GPLVM"
"STD_GPLVM"


class OE_GPLVM:
    def __init__(self, seed, model_name, tune=False):
        self.seed = seed
        self.model_name = model_name
        self.tune = tune
        self.batch_size = 128
        self.n_epochs = 5000
        self.likelihood = GaussianLikelihood()
        self.llkn = []
        self.klxn = []
        self.klun = []
        self.llka = []
        self.klxa = []
        self.klua = []
        self.loss_n = []
        self.loss_a = []
        self.elbo_max = []
        self.constant = []
        self.loss_total = []

    def _get_indices(self, y_train):
        idx_a = np.where(y_train == 1)[0]
        idx_n = np.where(y_train == 0)[0]
        self.ratio = len(idx_a) / (len(idx_a) + len(idx_n))
        qtd_anomaly = int(self.ratio * self.batch_size)
        qtd_normal = self.batch_size - qtd_anomaly
        idx_n = torch.tensor(np.random.choice(idx_n, qtd_normal, replace=True))

        if qtd_anomaly == 0:
            idx_a = torch.tensor(np.random.choice(idx_n, qtd_anomaly, replace=True))
        else:
            idx_a = torch.tensor(np.random.choice(idx_a, qtd_anomaly, replace=True))

        batch_index = torch.cat([idx_n, idx_a])

        return idx_n, idx_a, batch_index

    def _expected_log_prob(self, target):  # X, indices):
        output = self.model(self.model.sample_latent_variable(target))
        exp_log_prob = self.likelihood.expected_log_prob(target.T, output)
        return exp_log_prob

    def _kl_divergence_latent(self, target):  # X, indices=None):
        mu, sigma, local_q_x = create_dist_qx(self.model, target)
        local_p_x = create_dist_prior(target, mu)

        return kl_divergence(local_q_x, local_p_x).div(self.n_train)

    def _kl_divergence_variational(self, target):
        ll_shape = torch.zeros_like(target.T)
        klu = (
            ll_shape.T.add_(
                self.model.variational_strategy.kl_divergence().div(self.batch_size)
            )
            .sum(-1)
            .T.div((self.n_train))
        )
        return klu

    def _get_loe_index(self, X, indices):
        ll_0, klu_0, kl_0 = self._calculate_terms(X, indices)
        score = ll_0 - kl_0

        qtd_normal = int(score.shape[0] * (1 - self.ratio))
        qtd_anormal = self.batch_size - int(score.shape[0] * (1 - self.ratio))

        _, loe_idx_n = torch.topk(score, qtd_normal, largest=True, sorted=False)
        _, loe_idx_a = torch.topk(score, qtd_anormal, largest=False, sorted=False)
        return indices[loe_idx_n], indices[loe_idx_a]

    def _calculate_terms(self, X, indices):
        batch_target = X[indices]
        mu, sigma, local_q_x = create_dist_qx(self.model, batch_target)
        local_p_x = create_dist_prior(batch_target, mu)
        batch_output = self.model(self.model.sample_latent_variable(batch_target))
        exp_log_prob = self.likelihood.expected_log_prob(batch_target.T, batch_output)
        log_likelihood = exp_log_prob.sum(0).div(self.batch_size)  # Vetor 1xN
        kl_x = kl_divergence(local_q_x, local_p_x).div(self.n_train)  # Vetor 1xN
        kl_u = self._kl_divergence_variational(batch_target)  # Vetor 1xN

        #log_marginal = self.likelihood.log_marginal(batch_target.T, batch_output).sum(0).div(self.batch_size)

        return log_likelihood, kl_u, kl_x

    def loss_fn(self, X_train, batch_index):
        self.elbo_term = 0
        self.elbo_term_max = 0
        ll, klu, klx = self._calculate_terms(X_train, batch_index)
        elbo = ll - klu - klx
        self.elbo_term = elbo
        self.elbo_term_max = elbo.max()
        loss_normal = -elbo
        loss_anomaly = -torch.log(
            0.001 + torch.exp(elbo).max().item() - torch.exp(elbo)
        )
        return loss_normal, loss_anomaly

    def fit(self, X_train, y_train, ratio=False):
        self.n_train = len(X_train)
        self.data_dim = X_train.shape[1]

        if self.tune:
            pass
        else:
            self.nn_layers = (5, 5)
            self.n_inducing = 50
            self.latent_dim = 2
            self.lr = 0.01
            self.alpha_n = 1
            self.alpha_a = -1

        X_prior_mean = torch.zeros(self.n_train, self.latent_dim)
        X_prior_covar = torch.eye(X_prior_mean.shape[1])
        prior_x = MultivariateNormalPrior(X_prior_mean, X_prior_covar)
        self.encoder = NNEncoder(
            self.n_train, self.latent_dim, prior_x, self.data_dim, self.nn_layers
        )
        self.model = AEB_GPLVM(
            self.n_train,
            self.data_dim,
            self.latent_dim,
            self.n_inducing,
            self.encoder,
            self.nn_layers,
        )

        

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            self.lr,
        )

        #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.8)

        self.model.train()
        self.abs_diff = []
        iterator = trange(self.n_epochs, leave=True)


        for i in iterator:
            self.optimizer.zero_grad()
            _, _, batch_index = self._get_indices(y_train)
            idx_n, idx_a = self._get_loe_index(X_train, batch_index)
            ll_n, klu_n, kl_n = self._calculate_terms(X_train, idx_n)
            ll_a, klu_a, kl_a = self._calculate_terms(X_train, idx_a)

            self.klun.append(klu_n.sum().detach().numpy())
            self.klua.append(klu_a.sum().detach().numpy())

            self.klxn.append(kl_n.sum().detach().numpy())
            self.klxa.append(kl_a.sum().detach().numpy())

            self.llkn.append(ll_n.sum().detach().numpy())
            self.llka.append(ll_a.sum().detach().numpy())

            abs_diff = ll_a.sum().detach().numpy() - (ll_n.sum().detach().numpy())
            self.abs_diff.append(abs_diff)

            
            loss_normal, loss_anomaly = (ll_n - klu_n - kl_n).sum(), (
                ll_a - klu_a - kl_a
            ).sum()

            self.loss = -(
                self.alpha_n * loss_normal + self.alpha_a * loss_anomaly
            ).sum()

            #if abs_diff < -0.001:
            #    self.loss = -(loss_normal + loss_anomaly).sum()

            self.loss.backward()
            self.optimizer.step()

            iterator.set_description(
                "Loss: " + str(float(np.round(self.loss.item(), 2))) + ", iter no: "
            )

            #if self.loss.item() < -1e-2:
            #    break

    def fit_loe(self, X_train, y_train, ratio=False):
        self.n_train = len(X_train)
        self.data_dim = X_train.shape[1]

        if self.tune:
            pass
        else:
            self.nn_layers = (5, 5)
            self.n_inducing = 50
            self.latent_dim = 2
            self.lr = 0.01
            self.alpha_n = 1
            self.alpha_a = -1

        X_prior_mean = torch.zeros(self.n_train, self.latent_dim)
        X_prior_covar = torch.eye(X_prior_mean.shape[1])
        prior_x = MultivariateNormalPrior(X_prior_mean, X_prior_covar)
        self.encoder = NNEncoder(
            self.n_train, self.latent_dim, prior_x, self.data_dim, self.nn_layers
        )
        self.model = AEB_GPLVM(
            self.n_train,
            self.data_dim,
            self.latent_dim,
            self.n_inducing,
            self.encoder,
            self.nn_layers,
        )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            self.lr,
        )

        self.model.train()
        iterator = trange(self.n_epochs, leave=True)

        for i in iterator:
            self.optimizer.zero_grad()
            _, _, batch_index = self._get_indices(y_train)
            loss_normal, loss_anomaly = self.loss_fn(X_train, batch_index)
            self.loss_n.append(loss_normal.mean().detach().numpy())
            self.loss_a.append(loss_anomaly.mean().detach().numpy())

            score = loss_normal - loss_anomaly

            _, idx_n = torch.topk(
                score,
                int(score.shape[0] * (1 - self.ratio)),
                largest=False,
                sorted=False,
            )
            _, idx_a = torch.topk(
                score, int(score.shape[0] * self.ratio), largest=True, sorted=False
            )
            loss = torch.cat(
                [
                    loss_normal[idx_n],
                    0.5 * loss_normal[idx_a] + 0.5 * loss_anomaly[idx_a],
                ],
                0,
            )
            loss_mean = -(-loss.mean())

            #if i <= 1000:
            #    loss = loss_normal
            #    loss_mean = loss_normal.mean()

            # self.loss = -(
            #    self.alpha_n * loss_normal + self.alpha_a * loss_anomaly
            # ).sum()
            # self.loss.backward()
            # self.optimizer.zero_grad()
            loss_mean.backward()
            self.optimizer.step()

            self.loss_total.append(loss_mean.item())
            iterator.set_description(
                "Loss: " + str(float(np.round(loss_mean.item(), 2))) + ", iter no: "
            )
            # if self.loss.item() < -1e8:
            #   break

        pass
    def fit_loe_2(self, X_train, y_train, ratio=False):
        self.n_train = len(X_train)
        self.data_dim = X_train.shape[1]

        if self.tune:
            pass
        else:
            self.nn_layers = (5, 5)
            self.n_inducing = 50
            self.latent_dim = 2
            self.lr = 0.01
            self.alpha_n = 1
            self.alpha_a = -1

        X_prior_mean = torch.zeros(self.n_train, self.latent_dim)
        X_prior_covar = torch.eye(X_prior_mean.shape[1])
        prior_x = MultivariateNormalPrior(X_prior_mean, X_prior_covar)
        self.encoder = NNEncoder(
            self.n_train, self.latent_dim, prior_x, self.data_dim, self.nn_layers
        )
        self.model = AEB_GPLVM(
            self.n_train,
            self.data_dim,
            self.latent_dim,
            self.n_inducing,
            self.encoder,
            self.nn_layers,
        )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            self.lr,
        )

        self.model.train()
        iterator = trange(self.n_epochs, leave=True)
        for i in iterator:
            self.optimizer.zero_grad()
            _, _, batch_index = self._get_indices(y_train)
            idx_n, idx_a = self._get_loe_index(X_train, batch_index)
            ll_n, klu_n, kl_n = self._calculate_terms(X_train, idx_n)
            ll_a, klu_a, kl_a = self._calculate_terms(X_train, idx_a)

            self.klun.append(klu_n.sum().detach().numpy())
            self.klua.append(klu_a.sum().detach().numpy())

            self.klxn.append(kl_n.sum().detach().numpy())
            self.klxa.append(kl_a.sum().detach().numpy())

            self.llkn.append(ll_n.sum().detach().numpy())
            self.llka.append(ll_a.sum().detach().numpy())
            loss_normal, loss_anomaly = (ll_n - klu_n - kl_n).sum(), (
                ll_a - klu_a - kl_a
            ).sum()

            self.loss = -(
                self.alpha_n * loss_normal + self.alpha_a * loss_anomaly
            ).sum()
            self.loss.backward()
            self.optimizer.step()

            iterator.set_description(
                "Loss: " + str(float(np.round(self.loss.item(), 2))) + ", iter no: "
            )
            if self.loss.item() < -1e8:
                break
    def predict_score(self, X_test):
        with torch.no_grad():
            self.model.eval()
            self.likelihood.eval()
        n_test = len(X_test)
        mu, sigma, local_q_x = create_dist_qx(self.model, X_test)
        local_p_x = create_dist_prior(X_test, mu)
        X_pred = self.model(self.model.sample_latent_variable(X_test))
        exp_log_prob = self.likelihood.expected_log_prob(X_test.T, X_pred)
        log_likelihood = exp_log_prob.sum(0).div(n_test)
        kl_x = kl_divergence(local_q_x, local_p_x).div(n_test)
        kl_u = self._kl_divergence_variational(X_test)
        score = -(log_likelihood - kl_u - kl_x).detach().numpy()
        score = MinMaxScaler().fit_transform(np.reshape(score, (-1, 1)))
        return score


class STD_GPLVM:
    def __init__(self, seed, model_name, tune=False):
        pass

    def fit(self, X_train, y_train, ratio):
        pass

    def predict_score(self, X_test):
        pass


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


def calculate_elbo(
    model, likelihood, target, num_data, batch_size, elbo_shape="default"
):
    batch_target = target
    mu, sigma, local_q_x = create_dist_qx(model, batch_target)
    local_p_x = create_dist_prior(batch_target, mu)
    batch_output = model(model.sample_latent_variable(batch_target))
    exp_log_prob = likelihood.expected_log_prob(batch_target.T, batch_output)

    if elbo_shape == "default":
        # Isso aqui acaba sendo a mesma implementação do GpyTorch
        # VariationalELBO(..., combine_terms = False)
        local_dim_data = batch_target.shape[1]
        log_likelihood = exp_log_prob.sum(-1).div(batch_size)  # Vetor 1xD
        kl_term = KLGaussianAddedLossTerm(
            local_q_x, local_p_x, num_data, local_dim_data
        )
        kl_u = model.variational_strategy.kl_divergence().div(num_data)  # Vetor 1xD
        kl_x = torch.zeros_like(log_likelihood).add(kl_term.loss())  # Vetor 1xD

        return log_likelihood, kl_u, kl_x
    elif elbo_shape == "loe":
        log_likelihood = exp_log_prob.sum(0).div(batch_size)  # Vetor 1xN
        kl_x = kl_divergence(local_q_x, local_p_x).div(num_data)  # Vetor 1xN
        kl_u = 0
        return log_likelihood, kl_u, kl_x


def get_loe_idx(model, likelihood, Y_train, batch_index, train_data, ratio):
    batch_train = Y_train[batch_index]
    batch_size = len(batch_index)

    ll_0, klu_0, kl_0 = calculate_elbo(
        model, likelihood, batch_train, batch_size, train_data, elbo_shape="loe"
    )
    score = ll_0 - kl_0

    qtd_normal = int(score.shape[0] * (1 - ratio))
    qtd_anormal = batch_size - int(score.shape[0] * (1 - ratio))

    _, loe_idx_n = torch.topk(score, qtd_normal, largest=True, sorted=False)
    _, loe_idx_a = torch.topk(score, qtd_anormal, largest=False, sorted=False)

    return batch_index[loe_idx_n], batch_index[loe_idx_a]


def loss_loe(method, loss_normal, loss_anomaly):
    if method in ["blind", "refine"]:
        return (loss_normal + loss_anomaly).sum()
    elif method == "not_so_blind":
        return (2 * loss_normal + loss_anomaly).sum()
    elif method == "soft":
        return (0.75 * loss_normal + (-0.25) * loss_anomaly).sum()
    elif method == "hard":
        return (loss_normal + (-1) * loss_anomaly).sum()


def get_batch_indices(batch_size, labels, method=None):
    idx_a = np.where(labels == 1)[0]
    idx_n = np.where(labels == 0)[0]
    ratio = len(idx_a) / (len(idx_a) + len(idx_n))
    if method == "refine":
        return torch.tensor(
            np.random.choice(np.where(labels == 0)[0], batch_size, replace=True)
        )
    else:
        qtd_anomaly = int(ratio * batch_size)
        qtd_normal = batch_size - qtd_anomaly
        idx_n = torch.tensor(np.random.choice(idx_n, qtd_normal, replace=True))
        idx_a = torch.tensor(np.random.choice(idx_a, qtd_anomaly, replace=True))

        return torch.cat([idx_n, idx_a])
