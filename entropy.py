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

def prob_counts(counts, N):
    return [counts[x] / N for x in counts]

def mle(S, N, counts):
    return -sum([x * math.log(x) if x else 0 for x in S])

def miller_madow(S, N, counts):
    # N is the number of values sampled, m is the number of non-zero classes
    m = len(S)
    res = -sum([x * math.log(x) if x else 0 for x in S])
    return res + (m - 1) / (2 * N)

# def jackknife(S, N, counts):
#     res = N * mle(S, N, counts)
#     for i in counts:
#         counts[i] -= 1
#         S = [counts[x] / (N - 1) for x in counts]
#         mle_less = -sum([x * math.log(x) if x else 0 for x in S])
#         res -= ((N - 1) / N) * (counts[i] + 1) * mle_less
#         counts[i] += 1
#     return res

def jackknife(S, N, counts):
    res = N * mle(S, N, counts)
    S = [counts[x] / (N - 1) for x in counts]
    mle_less = -sum([x * math.log(x) if x else 0 for x in S])
    rem = 1 / (N - 1)
    for i in range(len(counts)):
        old = -S[i] * math.log(S[i]) if S[i] else 0
        S[i] -= rem
        new = -S[i] * math.log(S[i]) if S[i] else 0
        diff = new - old
        res -= ((N - 1) / N) * ((S[i] * (N - 1)) + 1) * (mle_less + diff)
        S[i] += rem
    return res

def horvitz_thompson(S, N, counts):
    return -sum([x * math.log(x) / (1 - (1 - x)**N) if x else 0 for x in S])

def chao_shen(S, N, counts):
    f_1 = list(counts.values()).count(1)
    if f_1 == N: f_1 -= 1
    C = 1 - (f_1 / N)
    return -sum([C * x * (math.log(C * x)) / (1 - (1 - C * x)**N) if C * x else 0 for x in S])

# http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-3.html
def wolpert_wolf(S, N, counts, α=1):
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

def nsb(S, N, counts):
    return ndd.entropy(counts)

# α / Kα = 1 / K
# α(Kα - α) / (α^2(α + 1)) = (Kα^2 - α^2) / (α^2 (α + 1)) = (K - 1) / (α + 1)
