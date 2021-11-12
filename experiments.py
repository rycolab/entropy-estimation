import entropy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
from collections import defaultdict
from scipy.stats import ttest_rel
from tqdm import tqdm
import plotnine as p9
import pandas as pd
from collections import Counter

estimators = ["MLE", "Horvitz-Thompson", "Chao-Shen", "Miller-Madow", "Jackknife", "NSB"]

np.random.seed(42)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

# root mean squared error
def rmse(samples):
    res = sum([(x[0] - x[1])**2 for x in samples])
    res = math.sqrt(res / len(samples))
    return res

def avgbias(samples):
    res = sum([(x[0] - x[1]) for x in samples])
    res = res / len(samples)
    return res

def permutation_test(p, q, perms=10000, rmse=False):
    n = len(p)
    l = list(zip(p, q))
    diff = sum([(x[0] - x[1])**(2 if rmse else 1) for x in l]) / n
    print('Running permutation test')
    greater = 0
    diffs = []
    for i in tqdm(range(perms)):
        assign = [0 if x <= 0.5 else 1 for x in np.random.rand(len(p))]
        res = 0
        for pos, val in enumerate(assign):
            if val == 0:
                res += (l[pos][0] - l[pos][1])**(2 if rmse else 1)
            else:
                res += (l[pos][1] - l[pos][0])**(2 if rmse else 1)
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

def zipf(K=100):
    s = 0
    for i in range(1, K + 1):
        s += 1 / i

    # s * x = 1, normalise so x = 1 / s
    res = []
    for i in range(1, K + 1):
        res.append(1 / (i * s))
    return res

def mle_performance(epochs=10, sample_size=20, distrib_count=1000, K=100, zipf=False):
    funcs = [entropy.mle]
    # funcs = [entropy.mle, entropy.horvitz_thompson, entropy.chao_shen, entropy.miller_madow, entropy.jackknife, entropy.nsb]
    biases = [defaultdict(list) for x in range(len(funcs))]
    distribs = np.random.dirichlet([1] * K, distrib_count)
    # distribs = [zipf(K=K) for i in range(distrib_count)]

    for key, distrib in tqdm(enumerate(distribs), total=len(distribs)):
        true_entropy = -sum([x * math.log(x) for x in distrib])
        for i in range(1, K):
            distrib[i] += distrib[i - 1]
        x = []
        counts = Counter()
        N = 0
        for i in range(1, epochs + 1):
            add = np.random.rand(sample_size)
            x.append(i * sample_size)
            N += sample_size
            for obs in add:
                val = get_sample(distrib, obs) + 1
                counts[val] += 1
            S = entropy.prob_counts(counts, N)
            for num, func in enumerate(funcs):
                calc = func(S, N, counts)
                biases[num][i * sample_size].append((calc, true_entropy))

    dat = []
    for i in range(len(funcs)):
        data = list([[(y[0] - y[1]) for y in x] for x in biases[i].values()])
        data = list(zip(biases[i].keys(), data))
        data = [[[a, z, estimators[i]] for z in b] for a, b in data]
        data = sum(data, [])
        dat.extend(data)
    df = pd.DataFrame(dat, columns=['Samples', 'Bias (nats)', 'Estimator'])
    graph = (p9.ggplot(data=df, mapping=p9.aes(x='Samples', y='Bias (nats)', group='Samples', fill='factor(Estimator)'))
        + p9.geom_boxplot(width=sample_size * 0.8, show_legend=False, outlier_alpha=0.1)
        # + p9.geom_violin(width=sample_size * 0.8, show_legend=False)
        + p9.facet_wrap('~Estimator') + p9.theme(text=p9.themes.element_text(family='serif'))
        + p9.labels.ggtitle('MLE bias (Symmetric Dirichlet, $K=100$)'))
        # + p9.labels.ggtitle('Sample size vs. estimator bias (Zipfian, $K=100$)'))
    graph.draw()
    graph.save('figures/mle_bias.pdf', width=3.5, height=2.75)
    plt.show()

# symmetric dirichlet
def symmetric(fout, epochs=1, sample_size=1000, distrib_count=10000, K=2, samples=None):
    if not samples:
        samples = [sample_size] * epochs
    funcs = [entropy.mle, entropy.horvitz_thompson, entropy.chao_shen, entropy.miller_madow, entropy.jackknife, entropy.nsb]
    distribs = np.random.dirichlet([1] * K, distrib_count)
    # distribs = [zipf(K=K)] * distrib_count

    if K == 2 and distrib_count==50:
        plt.xlim(right=1)
        plt.barh(range(distrib_count), distribs.transpose()[0])
        plt.barh(range(distrib_count), distribs.transpose()[1], left=distribs.transpose()[0])
        plt.title("Sample distributions (Dirichlet, $K = 2$)")
        plt.savefig("figures/distributions.pdf")
        plt.clf()

    biases = [defaultdict(list) for x in range(len(funcs))]

    # generate samples and get entropy estimates
    for key, distrib in tqdm(enumerate(distribs), total=len(distribs)):
        true_entropy = -sum([x * math.log(x) for x in distrib])
        for i in range(1, K):
            distrib[i] += distrib[i - 1]
        X = []
        counts = Counter()
        N = 0
        for sample in samples:
            add = np.random.rand(sample)
            N += sample
            X.append(N)
            for obs in add:
                val = get_sample(distrib, obs) + 1
                counts[val] += 1
            S = entropy.prob_counts(counts, N)
            for num, func in enumerate(funcs):
                calc = func(S, N, counts)
                biases[num][N].append((calc, true_entropy))

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
    tot = 0
    for n in X:
        tot += n
        fout.write(f'N={tot}, K={K}\n')
        for num in range(len(funcs)):
            bias = avgbias(biases[num][n])
            bias2 = rmse(biases[num][n])
            print(f'{estimators[num]}: bias <{bias}>, rmse <{bias2}>')
            fout.write(f'{estimators[num]}: bias <{bias}>, rmse <{bias2}>\n')

        # permutation test against true
        for num in range(len(funcs)):
            a, b = [x[0] for x in biases[num][n]], [x[1] for x in biases[num][n]]
            res, diff = permutation_test(a, b, rmse=True)
            fout.write(f'{estimators[num]} vs. True: greater <{res}>, diff <{diff}>\n')

        # permutation test pairwise
        for num1 in range(len(funcs)):
            for num2 in range(num1 + 1, len(funcs)):
                n = list(biases[num1].keys())[-1]
                a, b = [x[0] for x in biases[num1][n]], [x[0] for x in biases[num2][n]]
                res, diff = permutation_test(a, b, rmse=True)
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
            symmetric(fout, samples=[10, 90, 900, 9000], distrib_count=1000, K=K)