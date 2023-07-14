#!/usr/bin/env python3

from abc import ABC, abstractmethod

import torch

from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood


class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood, ABC):
    def __init__(self, likelihood, model, num_data, beta=1.0, combine_terms=True):
        super().__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data
        self.beta = beta

    @abstractmethod
    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        approximate_dist_n,
        target_n,
        approximate_dist_a,
        target_a,
        method,
        **kwargs
    ):

        num_batch = (
            approximate_dist_n.event_shape[0] + approximate_dist_a.event_shape[0]
        )

        log_likelihood_n = self._log_likelihood_term(
            approximate_dist_n, target_n, **kwargs
        )

        log_likelihood_a = self._log_likelihood_term(
            approximate_dist_a, target_a, **kwargs
        )

        if method in ["blind", "refine"]:
            log_likelihood = (log_likelihood_n + log_likelihood_a).div(num_batch)
        elif method == "hard":
            log_likelihood = (log_likelihood_n - log_likelihood_a).div(num_batch)
        elif method == "soft":
            log_likelihood = (
                log_likelihood_n - 0.25 * (log_likelihood_a + log_likelihood_n)
            ).div(num_batch)
        else:
            raise NotImplementedError("Metodo nao existente")

        kl_divergence = self.model.variational_strategy.kl_divergence().div(
            self.num_data / self.beta
        )

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True
        
        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss
        else: 
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior


class ComposedVariationalELBO(_ApproximateMarginalLogLikelihood):
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        return self.likelihood.expected_log_prob(
            target, variational_dist_f, **kwargs
        ).sum(-1)

    def forward(
        self,
        variational_dist_n,
        target_n,
        variational_dist_a,
        target_a,
        method,
        **kwargs
    ):
        return super().forward(
            variational_dist_n, target_n, variational_dist_a, target_a, method, **kwargs
        )
