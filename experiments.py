import entropy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
from collections import defaultdict
from scipy.stats import ttest_rel
from tqdm import tqdm

estimators = ["MLE", "Horvitz-Thompson", "Chao-Shen", "Miller-Madow", "Jackknife", "NSB"]

np.random.seed(42)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (4,3)
plt.rc('font', family='Helvetica')

# root mean squared error
def rmse(samples):
    res = sum([(x[0] - x[1])**2 for x in samples])
    res = math.sqrt(res / len(samples))
    return res

def avgbias(samples):
    res = sum([(x[0] - x[1]) for x in samples])
    res = res / len(samples)
    return res

def permutation_test(p, q, perms=10000):
    n = len(p)
    l = list(zip(p, q))
    diff = sum([x[0] - x[1] for x in l]) / n
    print('Running permutation test')
    greater = 0
    diffs = []
    for i in tqdm(range(perms)):
        assign = [0 if x <= 0.5 else 1 for x in np.random.rand(len(p))]
        res = 0
        for pos, val in enumerate(assign):
            if val == 0:
                res += l[pos][0] - l[pos][1]
            else:
                res += l[pos][1] - l[pos][0]
        res /= n
        diffs.append(res)
        if res >= diff:
            greater += 1
    # plt.hist(diffs)
    # plt.axvline(x=diff)
    # plt.show()
    return greater / perms, diff

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

def mle_performance(epochs=20, sample_size=20, distrib_count=1000, K=100):
    funcs = [entropy.mle]
    biases = [defaultdict(list) for x in range(len(funcs))]
    distribs = np.random.dirichlet([1] * K, distrib_count)
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
            S, N, counts = entropy.prob(sample)
            for num, func in enumerate(funcs):
                biases[num][i * sample_size].append((func(S, N, counts), true_entropy))

    plt.rcParams['figure.figsize'] = (10,7)
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    plt.title("MLE entropy estimates for symmetric Dirichlet ($K = 100$)")
    plt.ylabel("Bias from true entropy (nats)")
    plt.xlabel("$N$, sample size")
    plt.xticks(rotation = 45)
    ax.boxplot(list([[(y[0] - y[1]) for y in x] for x in biases[0].values()]), positions=list(biases[0].keys()), widths=sample_size * 0.8, showmeans=True)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.savefig("figures/mle_bias.pdf")
    plt.show()

# symmetric dirichlet
def symmetric(fout, epochs=1, sample_size=1000, distrib_count=10000, K=2):

    funcs = [entropy.mle, entropy.horvitz_thompson, entropy.chao_shen, entropy.miller_madow, entropy.jackknife, entropy.nsb]
    distribs = np.random.dirichlet([1] * K, distrib_count)

    if K == 2 and distrib_count==50:
        plt.xlim(right=1)
        plt.barh(range(distrib_count), distribs.transpose()[0])
        plt.barh(range(distrib_count), distribs.transpose()[1], left=distribs.transpose()[0])
        plt.title("Sample distributions (Dirichlet, $K = 2$)")
        plt.savefig("figures/distributions.pdf")
        plt.clf()

    biases = [defaultdict(list) for x in range(len(funcs))]

    # generate samples and get entropy estimates
    for key, distrib in tqdm(enumerate(distribs), total=distrib_count):
        true_entropy = -sum([x * math.log(x) for x in distrib])
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
            S, N, counts = entropy.prob(sample)
            for num, func in enumerate(funcs):
                biases[num][i * sample_size].append((func(S, N, counts), true_entropy))
            # print(biases)
            # input()
            # entropies.append(entropy.mle(sample))

    # # paired t-test
    # dists = []
    # for num in range(len(funcs)):
    #     print(estimators[num])
    #     n = list(biases[num].keys())[-1]
    #     # print([x[0] for x in biases[num][n]], [x[1] for x in biases[num][n]])
    #     bias = avgbias(biases[num][n])
    #     dists.append([x[0] - x[1] for x in biases[num][n]])
    #     ttest = ttest_rel([x[0] for x in biases[num][n]], [x[1] for x in biases[num][n]])
    #     print(n, bias, ttest)
    # plt.boxplot(dists)
    # plt.show()
    # plt.clf()

    # average bias
    for num in range(len(funcs)):
        n = list(biases[num].keys())[-1]
        bias = avgbias(biases[num][n])
        bias2 = rmse(biases[num][n])
        fout.write(f'{estimators[num]}: bias <{bias}>, rmse <{bias2}>\n')

    # permutation test
    for num1 in range(len(funcs)):
        for num2 in range(num1 + 1, len(funcs)):
            n = list(biases[num1].keys())[-1]
            a, b = [x[0] for x in biases[num1][n]], [x[0] for x in biases[num2][n]]
            res, diff = permutation_test(a, b)
            fout.write(f'{estimators[num1]} vs. {estimators[num2]}: greater <{res}>, diff <{diff}>\n')

    # plot bias curve
    if epochs != 1:
        l = len(biases[0].values())
        plt.rcParams['figure.figsize'] = (10,7)
        plt.rcParams.update({'font.size': 15})
        for num in range(len(funcs)):
            print(num)
            plt.plot(list(biases[num].keys()), [avgbias(x) for x in biases[num].values()])
        plt.legend(estimators)
        plt.show()

if __name__ == '__main__':
    # mle_performance()
    with open('logs/symmetric.txt', 'w') as fout:
        for K in [2, 5, 10, 100, 1000]:
            for N in [10, 100, 1000, 10000]:
                fout.write(f'K = {K}, N = {N}\n')
                symmetric(fout, epochs=1, sample_size=N, distrib_count=1000, K=K)