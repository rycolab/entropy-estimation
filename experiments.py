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
import os
import pickle

estimators = ["MLE", "HT", "CS", "MM", "J", "NSB"]
funcs = [entropy.mle, entropy.horvitz_thompson, entropy.chao_shen, entropy.miller_madow, entropy.jackknife, entropy.nsb]

np.random.seed(42)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

# mean squared error
def get_mse(samples):
    res = sum([(x[0] - x[1])**2 for x in samples])
    res = res / len(samples)
    return res

def get_bias(samples):
    res = sum([(x[0] - x[1]) for x in samples])
    res = res / len(samples)
    return res

def get_mab(samples):
    res = sum([abs(x[0] - x[1]) for x in samples])
    res = res / len(samples)
    return res

def permutation_test(p, q, true, num1, num2, perms=10000):
    n = len(p)
    l = list(zip(p, q, true))
    l = [[x[0] - x[2], x[1] - x[2], x[2]] for x in l]
    # bias of p, bias of q, true
    
    # difference in MAB = 1/n Σ abs(pᵢ - trueᵢ) - 1/n Σ abs(qᵢ - trueᵢ)
    mab = sum([abs(x[0]) for x in l]) / n - sum([abs(x[1]) for x in l]) / n
    # difference in MSE = 1/n Σ (pᵢ - trueᵢ)² - 1/n Σ (qᵢ - trueᵢ)²
    mse = sum([x[0]**2 for x in l]) / n - sum([x[1]**2 for x in l]) / n

    print('Running permutation test')
    greater_mab = 0
    greater_mse = 0

    mabs = []
    for i in tqdm(range(perms - 1)):
        # swap or not, binary array
        assign = [0 if x <= 0.5 else 1 for x in np.random.rand(len(p))]
        # get new mab and mse differences
        mab1, mab2 = 0, 0
        mse1, mse2 = 0, 0
        for pos, val in enumerate(assign):
            if val == 0:
                mab1 += abs(l[pos][0])
                mab2 += abs(l[pos][1])
                mse1 += l[pos][0]**2
                mse2 += l[pos][1]**2
            else:
                mab1 += abs(l[pos][1])
                mab2 += abs(l[pos][0])
                mse1 += l[pos][1]**2
                mse2 += l[pos][0]**2
        # check if diff in mabs/mses is greater than true
        mabs.append((mab1 - mab2) / n)
        if (mab1 - mab2) / n >= mab:
            greater_mab += 1
        if (mse1 - mse2) / n >= mse:
            greater_mse += 1

    # plt.hist(mabs)
    # plt.axvline(x=mab)
    # plt.title(f'{estimators[num1]} vs {estimators[num2]}')
    # plt.show()

    if greater_mab / perms < 0.5:
        greater_mab += 1
    if greater_mse / perms < 0.5:
        greater_mse += 1
    return greater_mab / perms, greater_mse / perms

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

def mle_performance(epochs=10, sample_size=20, distrib_count=100, K=100):
    # funcs = [entropy.mle]
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
        + p9.geom_boxplot(width=sample_size * 0.8, show_legend=False, outlier_alpha=0.0, size=0.2)
        # + p9.geom_violin(width=sample_size * 0.8, show_legend=False)
        + p9.facet_wrap('~Estimator', nrow=1) + p9.theme(text=p9.themes.element_text(family='serif'))
        # + p9.labels.ggtitle('MLE bias (Symmetric Dirichlet, $K=100$)'))
        + p9.labels.ggtitle('Estimator bias')
        + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5)))
        # + p9.labels.ggtitle('Sample size vs. estimator bias (Zipfian, $K=100$)'))
    graph.draw()
    graph.save('figures/estimators.pdf', width=8, height=1.3)
    plt.show()

def gigaword(fout):
    # funcs = [entropy.mle, entropy.horvitz_thompson, entropy.chao_shen, entropy.miller_madow, entropy.jackknife, entropy.nsb]
    biases = [defaultdict(list) for x in range(len(funcs))]

    X = []
    counts = Counter()
    N = 0
    
    # parse gigaword
    os.system('ls')
    with open('data/gigaword/gigaword.txt') as files:
        for file in tqdm(files):
            file = file.strip()
            os.system(f'curl https://gigaword.library.arizona.edu/data/xml/{file} -O')
            os.system(f'cd data/gigaword/agiga_1.0 && java -cp build/agiga-1.0.jar:lib/* edu.jhu.agiga.AgigaPrinter lemmas ../../{file} > ../../gigaword.data.txt')
            # data = os.popen(f'cd data/gigaword/agiga_1.0 && java -cp build/agiga-1.0.jar:lib/* edu.jhu.agiga.AgigaPrinter words ../../cna_eng_200307.xml.gz').read()
            with open('gigaword.data.txt', 'r') as fin:
                for sentence in fin:
                    for word in sentence.rstrip().split(' '):
                        if word == '': continue
                        counts[word] += 1
                        N += 1
                        if N % 1000 == 0:
                            X.append(N)
                            S = entropy.prob_counts(counts, N)
                            for num, func in enumerate(funcs):
                                calc = func(S, N, counts)
                                biases[num][N].append(calc)
                            print(N, biases[0][N])
                            with open('data/gigaword/entropies.pickle', 'wb') as handle:
                                pickle.dump(biases, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.remove(f'{file}')

        X.append(N)
        S = entropy.prob_counts(counts, N)
        for num, func in enumerate(funcs):
            calc = func(S, N, counts)
            biases[num][N].append(calc)
            print(estimators[num], calc)

# symmetric dirichlet
def symmetric(fout, epochs=1, sample_size=1000, distrib_count=10000, K=2, samples=None):
    if not samples:
        samples = [sample_size] * epochs
    # funcs = [entropy.mle, entropy.horvitz_thompson, entropy.chao_shen, entropy.miller_madow, entropy.jackknife, entropy.nsb]
    # distribs = np.random.dirichlet([1] * K, distrib_count)
    distribs = [zipf(K=K) for i in range(distrib_count)]

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
    #     bias = get_bias(biases[num][n])
    #     dists.append([x[0] - x[1] for x in biases[num][n]])
    #     ttest = ttest_rel([x[0] for x in biases[num][n]], [x[1] for x in biases[num][n]])
    #     print(n, bias, ttest)
    # plt.boxplot(dists)
    # plt.show()
    # plt.clf()

    # average bias
    # https://en.wikipedia.org/wiki/Bonferroni_correction
    alpha = 0.05 / (((len(estimators) - 1) * len(estimators)) / 2)
    print(f'alpha = {alpha}')
    fout.write(f'alpha = {alpha}\n')
    for n in X:
        fout.write(f'N={n}, K={K}\n')
        for num in range(len(funcs)):
            bias = get_bias(biases[num][n])
            mab = get_mab(biases[num][n])
            mse = get_mse(biases[num][n])
            print(f'{estimators[num]}: bias <{bias}>, mab <{mab}, mse <{mse}>')
            fout.write(f'{estimators[num]}: bias <{bias}>, mab <{mab}, mse <{mse}>\n')

        # # permutation test against true
        # for num in range(len(funcs)):
        #     a, b = [x[0] for x in biases[num][n]], [x[1] for x in biases[num][n]]
        #     mab, mse = permutation_test(a, b, rmse=True)
        #     fout.write(f'{estimators[num]} vs. True: greater <{res}> ({greater}), diff <{diff}>, significant <{res < alpha or res > (1- alpha)}>\n')

        # permutation test pairwise
        for num1 in range(len(funcs)):
            for num2 in range(num1 + 1, len(funcs)):
                a, b = [x[0] for x in biases[num1][n]], [x[0] for x in biases[num2][n]]
                true = [x[1] for x in biases[num1][n]]
                mab, mse = permutation_test(a, b, true, num1, num2)
                print(f'{estimators[num1]} vs. {estimators[num2]}: greater mab <{mab}> ({mab < alpha or mab > (1- alpha)}), greater mse <{mse}> ({mse < alpha or mse > (1- alpha)})')
                fout.write(f'{estimators[num1]} vs. {estimators[num2]}: greater mab <{mab}> ({mab < alpha or mab > (1- alpha)}), greater mse <{mse}> ({mse < alpha or mse > (1- alpha)})\n')

        fout.write('\n\n')
    # plot bias curve
    # if epochs != 1:
    # l = len(biases[0].values())
    # plt.rcParams['figure.figsize'] = (10,7)
    # plt.rcParams.update({'font.size': 15})
    # for num in range(len(funcs)):
    #     print(num)
    #     plt.plot(list(biases[num].keys()), [get_bias(x) for x in biases[num].values()])
    # plt.legend(estimators)
    # plt.show()

    # dat = []
    # for i in range(len(funcs)):
    #     data = list([[(y[0] - y[1]) for y in x] for x in biases[i].values()])
    #     data = list(zip(biases[i].keys(), data))
    #     data = [[[a, z, estimators[i]] for z in b] for a, b in data]
    #     data = sum(data, [])
    #     dat.extend(data)
    # df = pd.DataFrame(dat, columns=['Samples', 'Bias (nats)', 'Estimator'])
    # graph = (p9.ggplot(data=df, mapping=p9.aes(x='Samples', y='Bias (nats)', group='Samples', fill='factor(Estimator)'))
    #     + p9.geom_boxplot(width=sample_size * 0.8, show_legend=False, outlier_alpha=0.1)
    #     + p9.facet_wrap('~Estimator') + p9.theme(text=p9.themes.element_text(family='serif'))
    #     + p9.labels.ggtitle('MLE Bias'))
    # graph.draw()
    # graph.save(f'figures/mle_bias_{K}.pdf', width=7, height=4)
    # plt.show()

if __name__ == '__main__':
    mle_performance()
    # with open('logs/gigaword.txt', 'w') as fout:
    #     gigaword(fout)
    # with open('logs/symmetric.txt', 'w') as fout:
    #     for K in [2, 5, 10, 100, 1000]:
    #         symmetric(fout, samples=[10, 90, 900, 9000], distrib_count=1000, K=K)
    # with open('logs/symmetric.txt', 'w') as fout:
    #     for K in [100, 1000, 10000]:
    #         symmetric(fout, samples=[10, 90, 900, 9000], distrib_count=1000, K=K)