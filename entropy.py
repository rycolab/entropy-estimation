import math
from collections import Counter

def prob(sample):
    counts = Counter(sample)
    return [counts[x] / len(sample) for x in counts], len(sample), counts

def mle(sample):
    S, N, counts = prob(sample)
    return -sum([x * math.log(x, 2) if x else 0 for x in S])

def miller_madow(sample):
    # N is the number of values sampled, m is the number of non-zero classes
    S, N, counts = prob(sample)
    m = len(S)
    res = -sum([x * math.log(x, 2) if x else 0 for x in S])
    return res + (m - 1) / (2 * N)

# TODO: definitely a bug here lol
def jackknife(sample):
    S, N, counts = prob(sample)
    res = N * mle(sample)
    print(counts)
    for i in counts:
        counts[i] -= 1
        print(counts, counts[i] + 1)
        S = [counts[x] / len(sample) for x in counts]
        mle_less = -sum([x * math.log(x, 2) if x else 0 for x in S])
        res -= ((N - 1) / N) * (counts[i] + 1) * mle_less
        counts[i] += 1
    input()
    return res

def horvitz_thompson(sample):
    S, N, counts = prob(sample)
    return -sum([x * math.log(x, 2) / (1 - (1 - x)**N) if x else 0 for x in S])

def chao_shen(sample):
    S, N, counts = prob(sample)
    C = 1 - (list(counts.values()).count(1) / N)
    return -sum([C * x * math.log(C * x, 2) / (1 - (1 - C * x)**N) if x else 0 for x in S])