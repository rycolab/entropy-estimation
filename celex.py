import entropy
import plotnine as p9
import matplotlib.pyplot as plt
import numpy as np
from experiments import get_sample, permutation_test, funcs, get_bias, get_mab, get_mse
from collections import Counter
from tqdm import tqdm
import math
import pandas as pd

estimators = ["MLE", "HT", "CS", "MM", "J", "NSB"]

np.random.seed(87539319)
# https://en.wikipedia.org/wiki/Taxicab_number

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

def main(file, distribs=100, samples=[100, 900, 9000, 90000], perm=True, graph=False):
    fout = open(file, 'w')
    data = []
    langs = ['CELEX_V2/english/efw/efw.cd', 'CELEX_V2/german/gfw/gfw.cd',
        'CELEX_V2/dutch/dfw/dfw.cd', 'tagalog.txt', 'mongolian.txt']
    name = ['English', 'German', 'Dutch', 'Tagalog', 'Mongolian']
    for i, lang in enumerate(langs):
        print(lang)
        file = f'data/LDC96L14/{lang}'
        counts = Counter()
        N = 0
        with open(file, 'r') as fin:
            for row in fin:
                row = row.rstrip()
                if 'CELEX_V2' in file:
                    if '\\' not in row:
                        continue
                    _, word, lemma, freq = row.split('\\')[:4]
                else:
                    word, freq = row.split()
                freq = int(freq)
                if freq == 0:
                    continue
                counts[word] += freq
                N += freq
        S = entropy.prob_counts(counts, N)
        print(N, len(S))
        true_entropy = -sum([x * math.log(x) for x in S])
        for x in range(1, len(S)):
            S[x] += S[x - 1]
        
        
        # samples = [100, 9000, 90000]
        
        for trial in tqdm(range(distribs)):
            cur = Counter()
            n = 0
            for sample in samples:
                # generate 10k samples
                add = np.random.rand(sample)
                n += sample
                for obs in add:
                    val = get_sample(S, obs)
                    cur[val] += 1
                s = entropy.prob_counts(cur, n)
                mle = None
                for num, func in enumerate(funcs):
                    try:
                        calc = func(s, n, cur)
                        if abs(calc - true_entropy) > 100:
                            print(f'unstable estimate: {[n, calc - true_entropy, estimators[num], name[i]]}')
                            continue
                        data.append([n, calc - true_entropy, estimators[num], name[i]])
                    except Exception as e:
                        print(s, n)
                        print(cur.most_common(10))
                        raise e
        
        if perm:
            n = 0
            alpha = 0.05 / (((len(estimators) - 1) * len(estimators)) / 2)
            true = [true_entropy] * distribs
            for sample in samples:
                n += sample
                fout.write(f'{name[i]}, {n}\n')
                for num1, name1 in enumerate(estimators):
                    p = [x[1] for x in filter(lambda x: x[2] == name1 and x[0] == n and x[3] == name[i], data)]
                    bias = sum([x for x in p]) / distribs
                    mab = sum([abs(x) for x in p]) / distribs
                    mse = sum([abs(x)**2 for x in p]) / distribs
                    fout.write(f'{name1}: bias <{bias}>, mab <{mab}, mse <{mse}>\n')
                    # for num2, name2 in enumerate(estimators):
                    #     if num2 <= num1: continue
                    #     q = [x[1] for x in filter(lambda x: x[2] == name2 and x[0] == n and x[3] == name[i], data)]
                    #     print(len(p), len(q))
                    #     mab, mse = permutation_test(p, q, true, num1, num2)
                    #     fout.write(f'{name1} vs. {name2}: greater mab <{mab}> ({mab < alpha or mab > (1- alpha)}), greater mse <{mse}> ({mse < alpha or mse > (1- alpha)})\n')
                    #     print(name1, name2, mab, mse)
    fout.close()
        
    if graph:
        # for mse
        data = [[x[0], x[1]**2, x[2], x[3]] for x in data]
        df = pd.DataFrame(data, columns=['Samples', 'MSE (nats$^2$)', 'Estimator', 'Language'])
        graph = (p9.ggplot(data=df, mapping=p9.aes(x='Samples', y='MSE (nats$^2$)', color='Estimator'))
            # + p9.geom_boxplot(outlier_alpha=0.1, show_legend=False, width=1000 * 0.8)
            + p9.geom_hline(yintercept=0)
            + p9.geom_line()
            + p9.facet_wrap('~Language', nrow=2, ncol=3)
            + p9.scales.scale_x_log10()
            + p9.scales.scale_y_log10()
            + p9.theme(legend_title=p9.themes.element_text(size=0, alpha=0), 
                legend_text=p9.themes.element_text(size=7), axis_text_x=p9.element_text(rotation=90, hjust=0.5),
                legend_key_width=8, legend_key_height=10, legend_position=(0.8, 0.3))
            # + p9.labels.ggtitle(f'Entropy on CELEX')
            + p9.guides(color=p9.guide_legend(ncol=2)))
        graph.draw()
        graph.save('figures/celex_mse_big.pdf', width=3, height=2.5)
        plt.show()

if __name__ == '__main__':
    # main(file='logs/celex_2.txt', samples=[100000])
    # main(file='logs/celex_1.txt')
    #  + [10000] * 9 + [100000] * 9 + [1000000] * 4
    # 
    main(file='logs/celex_graph.txt', samples=[100] * 10 + [1000] * 9 + [10000] * 9 + [100000] * 9 + [1000000] * 4, distribs=1, perm=False, graph=True)
    # main(file='logs/celex_graph.txt', samples=[100, 900, 9000, 90000], distribs=10, perm=True, graph=False)