#!/usr/bin/env python
import math

import numpy as np
from scipy.optimize import minimize


# -------------------------------------------------------------------
# Beta-Binomial PMF and log-PMF
# -------------------------------------------------------------------
def beta_binomial_pmf(s, k, alpha, beta):
    """
    Beta-Binomial PMF: BB(s | k, alpha, beta) = C(k, s) * B(alpha+s, beta+k-s) / B(alpha, beta).
    """
    from math import exp, lgamma

    log_comb = math.lgamma(k + 1) - math.lgamma(s + 1) - math.lgamma(k - s + 1)
    log_num = (
        math.lgamma(alpha + s)
        + math.lgamma(beta + (k - s))
        - math.lgamma(alpha + beta + k)
    )
    log_den = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_p = log_comb + log_num - log_den
    return exp(log_p)


def log_beta_binomial_pmf(s, k, alpha, beta):
    """
    Returns the log of the Beta-Binomial PMF for s.
    """
    log_comb = math.lgamma(k + 1) - math.lgamma(s + 1) - math.lgamma(k - s + 1)
    log_num = (
        math.lgamma(alpha + s)
        + math.lgamma(beta + (k - s))
        - math.lgamma(alpha + beta + k)
    )
    log_den = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return log_comb + log_num - log_den


# -------------------------------------------------------------------
# EM for a 2-component mixture of Beta-Binomial distributions
# -------------------------------------------------------------------
def em_mixture_beta_binomial(
    counts: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42,
):
    """
    Fit a two-component mixture of Beta-Binomial distributions to observed counts {s_i},
    each s_i in [0, k].  The model is:
       S ~ w * BB(k, alpha1, beta1) + (1-w) * BB(k, alpha2, beta2).

    Returns a dictionary of the learned parameters:
        {
          'w': mixture weight,
          'alpha1': ...,
          'beta1': ...,
          'alpha2': ...,
          'beta2': ...,
          'log_likelihood': final log-likelihood,
          'n_iter': number of iterations
        }
    """
    rng = np.random.default_rng(random_state)

    # 1) Initialization
    w = 0.5
    alpha1, beta1 = 1.0 + 2 * rng.random(), 1.0 + 2 * rng.random()
    alpha2, beta2 = 1.0 + 2 * rng.random(), 1.0 + 2 * rng.random()

    # Define log-likelihood for the entire dataset
    def log_likelihood(params):
        w_, a1, b1, a2, b2 = params
        w_ = np.clip(w_, 1e-9, 1 - 1e-9)  # keep w in (0,1)
        ll = 0.0
        for s in counts:
            logp1 = log_beta_binomial_pmf(s, k, a1, b1)
            logp2 = log_beta_binomial_pmf(s, k, a2, b2)
            # log p = log( w * e^(logp1) + (1-w) * e^(logp2) )
            # do log-sum-exp for numerical stability
            c1 = np.log(w_) + logp1
            c2 = np.log(1 - w_) + logp2
            cmax = max(c1, c2)
            ll += cmax + math.log(math.exp(c1 - cmax) + math.exp(c2 - cmax))
        return ll

    def neg_log_likelihood(params):
        return -log_likelihood(params)

    old_ll = -np.inf
    n = len(counts)

    for iteration in range(max_iter):
        # E-step: compute responsibilities gamma_i = Prob(component=1 | s_i)
        logp1 = np.array([log_beta_binomial_pmf(s, k, alpha1, beta1) for s in counts])
        logp2 = np.array([log_beta_binomial_pmf(s, k, alpha2, beta2) for s in counts])

        logw1 = math.log(np.clip(w, 1e-9, 1 - 1e-9)) + logp1
        logw2 = math.log(np.clip(1 - w, 1e-9, 1 - 1e-9)) + logp2

        # denominator = log( e^(logw1) + e^(logw2) )
        max_ = np.maximum(logw1, logw2)
        denom = max_ + np.log(np.exp(logw1 - max_) + np.exp(logw2 - max_))
        gamma = np.exp(logw1 - denom)  # shape = (n,)

        # M-step: update w, alpha1, beta1, alpha2, beta2 by maximizing weighted LL
        w = gamma.mean()  # simple closed form for mixture weight

        # joint optimization over all five parameters
        # Our param vector is [w, alpha1, beta1, alpha2, beta2].
        x0 = [w, alpha1, beta1, alpha2, beta2]
        bnds = [
            (1e-9, 1 - 1e-9),  # w in (0,1)
            (1e-9, None),  # alpha1 > 0
            (1e-9, None),  # beta1 > 0
            (1e-9, None),  # alpha2 > 0
            (1e-9, None),  # beta2 > 0
        ]
        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bnds)
        w, alpha1, beta1, alpha2, beta2 = res.x
        w = np.clip(w, 1e-9, 1 - 1e-9)

        new_ll = log_likelihood([w, alpha1, beta1, alpha2, beta2])
        if abs(new_ll - old_ll) < tol:
            return {
                "w": w,
                "alpha1": alpha1,
                "beta1": beta1,
                "alpha2": alpha2,
                "beta2": beta2,
                "log_likelihood": new_ll,
                "n_iter": iteration + 1,
            }
        old_ll = new_ll

    # If max_iter reached
    return {
        "w": w,
        "alpha1": alpha1,
        "beta1": beta1,
        "alpha2": alpha2,
        "beta2": beta2,
        "log_likelihood": old_ll,
        "n_iter": max_iter,
    }
