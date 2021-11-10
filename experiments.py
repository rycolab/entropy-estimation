import entropy
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

np.random.seed(42)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (4,3)
plt.rc('font', family='Helvetica')

# root mean squared error
def rmse(samples):
    res = sum([(x[0] - x[1])**2 for x in samples])
    res = math.sqrt(res / len(samples))
    return res

# figure out which class was sampled using O(log K) binary search
def get_sample(arr, val):
    l, r = 0, len(arr) - 1
    while l <= r:
        m = (l + r) // 2
        if arr[m] >= val:
            r = m - 1
        else:
            l = m + 1
    return l

# symmetric dirichlet
def symmetric(epochs=100, sample_size=10, distrib_count=50, K=2):
    funcs = [entropy.mle, entropy.horvitz_thompson, entropy.chao_shen,
        entropy.miller_madow, entropy.jackknife, entropy.nsb]
    distribs = np.random.dirichlet([1] * K, distrib_count)

    if K == 2:
        plt.xlim(right=1)
        plt.barh(range(distrib_count), distribs.transpose()[0])
        plt.barh(range(distrib_count), distribs.transpose()[1], left=distribs.transpose()[0])
        plt.title("Sample distributions (Dirichlet, $K = 2$)")
        plt.savefig("distributions.pdf")
        plt.clf()

    rmses = [defaultdict(list) for x in range(len(funcs))]

    for key, distrib in enumerate(distribs):
        true_entropy = -sum([x * math.log(x) for x in distrib])
        print(key)
        for i in range(1, K):
            distrib[i] += distrib[i - 1]
        sample = []
        x = []
        # entropies = []
        for i in range(1, epochs + 1):
            add = np.random.rand(sample_size)
            x.append(i * sample_size)
            for obs in add:
                sample.append(get_sample(distrib, obs) + 1)
            for num, func in enumerate(funcs):
                rmses[num][i * sample_size].append((func(sample), true_entropy))
            # print(rmses)
            # input()
            # entropies.append(entropy.mle(sample))

    # plt.scatter(x, [true_entropy] * len(x))
    # plt.scatter(x, entropies)
    # print(rmses)
    l = len(rmses[0].values())
    # plt.boxplot(list([[y[0] - y[1] for y in x] for x in rmses[0].values()]), positions=list(range(l)))
    # plt.boxplot(list([[y[0] - y[1] for y in x] for x in rmses[1].values()]), positions=[x + 0.5 for x in range(l)])
    estimators = ["MLE", "Horvitz-Thompson", "Chao-Shen", "Miller-Madow", "Jackknife", "NSB"]
    for num in range(len(funcs)):
        print(num)
        plt.plot(list(rmses[num].keys()), [rmse(x) for x in rmses[num].values()])
    plt.legend(estimators)
    plt.show()

if __name__ == '__main__':
    symmetric(K=2)
    symmetric(K=10)
    symmetric(K=100)
    symmetric(K=1000)
    symmetric(K=10000)