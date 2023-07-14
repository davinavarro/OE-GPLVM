import torch
from gpytorch.mlls import KLGaussianAddedLossTerm
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import MultivariateNormalPrior
from torch.distributions import kl_divergence


def calculate_elbo(
    model, likelihood, target, num_data, batch_size, elbo_shape="default"
):
    batch_target = target

    # Criacao de Q_x
    mu = model.predict_latent(batch_target)[0]
    sigma = model.predict_latent(batch_target)[1]
    local_q_x = MultivariateNormal(mu, sigma)

    # Criacao da Prior
    local_batch_size = batch_target.shape[0]
    local_p_x_mean = torch.zeros(local_batch_size, mu.shape[1])
    local_p_x_covar = torch.eye(mu.shape[1])
    local_p_x = MultivariateNormalPrior(local_p_x_mean, local_p_x_covar)

    # Predicao do batch
    batch_output = model(model.sample_latent_variable(batch_target))

    # Expected Log Prob
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
        kl_x = torch.zeros_like(ll).add(kl_term.loss())  # Vetor 1xD

        return log_likelihood, kl_u, kl_x
    elif elbo_shape == "loe":
        log_likelihood = exp_log_prob.sum(0).div(batch_size)  # Vetor 1xN
        kl_x = kl_divergence(local_q_x, local_p_x).div(num_data)  # Vetor 1xN
        return log_likelihood, kl_x


def get_loe_idx(model, likelihood, Y_train, batch_index, train_data, ratio):
    batch_train = Y_train[batch_index]
    batch_size = len(batch_index)

    ll_0, kl_0 = calculate_elbo(
        model, likelihood, batch_train, batch_size, train_data, elbo_shape="loe"
    )
    score = ll_0 - kl_0

    qtd_normal = int(score.shape[0] * (1 - ratio))
    qtd_anormal = len(batch_index) - int(score.shape[0] * (1 - ratio))

    _, loe_idx_n = torch.topk(score, qtd_normal, largest=False, sorted=False)
    _, loe_idx_a = torch.topk(score, qtd_anormal, largest=True, sorted=False)

    return batch_index[loe_idx_n], batch_index[loe_idx_a]


def loss_loe(method, loss_normal, loss_anomaly):
    if method == "blind":
        return (loss_normal + loss_anomaly).sum()
    elif method == "soft":
        return (0.75 * loss_normal - 0.25 * loss_anomaly).sum()
    elif method == "hard":
        return (loss_normal - loss_anomaly).sum()
