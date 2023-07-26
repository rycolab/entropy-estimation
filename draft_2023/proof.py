from utils import lift, fsa_from_samples, estimate_entropy, get_samples, exact_entropy
from scipy.special import digamma, polygamma
from synthetic import make_acyclic_machine
import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    geom_line,
    geom_ribbon,
    aes,
    theme,
    element_text,
    facet_wrap,
    facet_grid,
    geom_histogram,
    geom_point
)
from plotnine.scales import scale_y_log10, scale_x_log10
from tqdm import tqdm

def var_ent(K: int=2):
    """Variance of entropy for a uniform Dirichlet distribution with K classes, per Archer (2014)"""
    if K <= 1:
        return 0.0
    
    exp_h_squared = 0.0
    exp_h_squared += ((K - 1) / (K + 1)) * ((digamma(2) - digamma(K + 2))**2                   - polygamma(1, K + 2))
    exp_h_squared += (      2 / (K + 1)) * ((digamma(3) - digamma(K + 2))**2 + polygamma(1, 3) - polygamma(1, K + 2))
    
    exp_h = digamma(K + 1) - digamma(2)
    # print(K, exp_h, -math.log(1 / K))

    return exp_h_squared - exp_h * exp_h

def acyclic_entropy_variance(K: int=2):
    """Compute exact variance of structured vs. unstructured MLE entropy for an acyclic FSA"""

    # unstructured
    paths = 2**(K - 2) if K >= 2 else 1
    unstructured = var_ent(int(paths))

    # structured
    structured = 0.0
    c = 1
    last_term = 1
    for i in range(K):
        arcs = K - i - 1
        if i >= 1:
            last_term = last_term / (K - i + 1)
            c += last_term
        print('    ', i, (c ** 2) * var_ent(arcs))
        structured += (c ** 2) * var_ent(arcs)
        if i == 0:
            c = 0
    
    return {
        "unstructured": unstructured,
        "structured": structured
    }

def acyclic_entropy_variance_monte_carlo(K: int=2, num_samps: int=10000):
    """Compute exact variance of structured vs. unstructured MLE entropy for an acyclic FSA"""
    vals = []
    for i in range(num_samps):
        fsa = make_acyclic_machine(states=K)
        res = exact_entropy(fsa)
        vals.append(res)
    print(f"{np.mean(vals):.10} {np.var(vals):.10}")

def dirichlet_entropy(alphas: list, K=None):
    if K is not None:
        A = K * alphas[0]
        res = digamma(A) + 1 / A
        res -= K * (alphas[0] / A) * digamma(alphas[0] + 1)
        return res
    else:
        A = sum(alphas)
        K = len(alphas)
        res = digamma(A) + 1 / A
        for alpha in alphas:
            res -= (alpha / A) * digamma(alpha + 1)
        return res

def dirichlet_squared_mean(alphas: list, K=None):

    # assert all alphas are equal
    if K is not None:
        A = K * alphas[0]
        res = 0.0
        I_ik = ((digamma(alphas[0] + 1) - digamma(A + 2)) * (digamma(alphas[0] + 1) - digamma(A + 2)) - polygamma(1, A + 2))
        res += (alphas[0] * alphas[0]) / ((A + 1) * A) * I_ik * (K**2 - K)
        J_i = ((digamma(alphas[0] + 2) - digamma(A + 2))**2 + polygamma(1, alphas[0] + 2) - polygamma(1, A + 2))
        res += (alphas[0] * (alphas[0] + 1)) / ((A + 1) * A) * J_i * K
        return res
    else:
        A = sum(alphas)
        K = len(alphas)
        res = 0.0
        for i in range(K):
            for k in range(K):
                if i != k:
                    I_ik = ((digamma(alphas[k] + 1) - digamma(A + 2)) * (digamma(alphas[i] + 1) - digamma(A + 2)) - polygamma(1, A + 2))
                    res += (alphas[i] * alphas[k]) / ((A + 1) * A) * I_ik
                else:
                    J_i = ((digamma(alphas[i] + 2) - digamma(A + 2))**2 + polygamma(1, alphas[i] + 2) - polygamma(1, A + 2))
                    res += (alphas[i] * (alphas[i] + 1)) / ((A + 1) * A) * J_i
        return res

def plot_entropy_vs_samples(K: int=2, alpha=1.0):
    ents = []
    for i in tqdm(range(1, 100)):
        ent = dirichlet_entropy([i / K + alpha] * K)
        ents.append([i, ent])

    df = pd.DataFrame(ents, columns=["samples", "entropy"])

    # plot
    p = (
        ggplot(df, aes(x="samples", y="entropy"))
        + geom_line()
        + geom_line(aes(x="samples", y=f"{np.log(K):.10}"), color="red")
    )
    print(p)

def plot_entropy_vs_classes(n: int=10):
    ents = []
    for k_incr in tqdm([x / 5 for x in range(5, 25)]):
        k = int(2**k_incr)
        for alp in [0.0, 1.0]:
            alpha = alp if alp != "1 / k" else 1.0 / k
            for samps_incr in [x / 5 for x in range(5, 100)]:
                for attempts in range(10):
                    samps = int(2**samps_incr)

                    # sample a distribution from dirichlet
                    s = np.random.dirichlet([0.5] * int(k), 1)[0]
                    s = s * samps
                    s += alpha

                    mean = dirichlet_entropy(s)
                    var = dirichlet_squared_mean(s) - mean**2
                    ents.append({
                        "classes": k,
                        "classes (2^k)": np.log(k) / np.log(2),
                        "entropy": mean,
                        "true": np.log(k),
                        "bias_squared": (mean - np.log(k))**2,
                        "bias": mean - np.log(k),
                        "variance": var,
                        "mse": (mean - np.log(k))**2 + var,
                        "N (2^k)": np.log(samps) / np.log(2),
                        "alpha": str(alp),
                    })

    df = pd.DataFrame(ents)
    df = pd.melt(df, id_vars=["classes (2^k)", "classes", "N (2^k)", "alpha"], value_vars=["bias_squared", "variance", "mse"])
    df = df.groupby(["classes (2^k)", "classes", "N (2^k)", "alpha", "variable"]).mean().reset_index()

    # plot
    p = (
        ggplot(df, aes(x="classes (2^k)", y="value", color="N (2^k)"))
        + geom_point()
        + facet_grid("variable~alpha")
        # + geom_line(aes(x="classes", y="true"), color="red")
    )
    # print(p)
    p.save("plots/entropy_vs_classes.png", dpi=300)

plot_entropy_vs_classes()