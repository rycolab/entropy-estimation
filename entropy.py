import math
from collections import Counter
from scipy.special import digamma, polygamma
from scipy import integrate, optimize
import numpy as np
import ndd

# all entropies are calculated in nats

def prob(sample):
    counts = Counter(sample)
    return [counts[x] / len(sample) for x in counts], len(sample), counts

def mle(sample):
    S, N, counts = prob(sample)
    return -sum([x * math.log(x) if x else 0 for x in S])

def miller_madow(sample):
    # N is the number of values sampled, m is the number of non-zero classes
    S, N, counts = prob(sample)
    m = len(S)
    res = -sum([x * math.log(x) if x else 0 for x in S])
    return res + (m - 1) / (2 * N)

def jackknife(sample):
    S, N, counts = prob(sample)
    res = N * mle(sample)
    for i in counts:
        counts[i] -= 1
        S = [counts[x] / (N - 1) for x in counts]
        mle_less = -sum([x * math.log(x) if x else 0 for x in S])
        res -= ((N - 1) / N) * (counts[i] + 1) * mle_less
        counts[i] += 1
    return res

def horvitz_thompson(sample):
    S, N, counts = prob(sample)
    return -sum([x * math.log(x) / (1 - (1 - x)**N) if x else 0 for x in S])

def chao_shen(sample):
    S, N, counts = prob(sample)
    C = 1 - (list(counts.values()).count(1) / N)
    if C == 0:
        C = 1
    return -sum([C * x * math.log(C * x) / (1 - (1 - C * x)**N) if x else 0 for x in S])

# http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-3.html
def wolpert_wolf(sample, α=1):
    S, N, counts = prob(sample)
    K = len(counts)
    α = 1
    res = digamma(N + K * α + 1)
    for i in counts:
        res -= ((counts[i] + α) / (N + K * α)) * digamma(counts[i] + α + 1)
    return res

# def nsb_α(K):
#     f = lambda u: (1 / math.log(K)) * (1/ u**2) * (1 / u - 1) * (K * polygamma(1, K / u - K + 1) - polygamma(1, 1 / u))
#     return integrate.quad(f, 0, 1)[0]

# def nsb_α(K):
#     p = lambda α: (1 / math.log(K)) * (K * polygamma(1, K * α + 1) - polygamma(1, α + 1))
#     integral = lambda x: integrate.quadrature(p, 0, x)[0] - 0.5
#     res = optimize.fsolve(integral, 1.0)
#     print(res, integrate.quad(p, 0, res))
#     return res

def nsb(sample):
    S, N, counts = prob(sample)
    return ndd.entropy(counts)

# α / Kα = 1 / K
# α(Kα - α) / (α^2(α + 1)) = (Kα^2 - α^2) / (α^2 (α + 1)) = (K - 1) / (α + 1)
