"""
Experiment on structured entropy prediction using Markov model of POS tags from UD.
"""

from os import stat
import glob
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
import sys
from concurrent.futures import ThreadPoolExecutor

from utils import lift, fsa_from_samples, estimate_entropy, get_samples

import conllu
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
from collections import defaultdict

from plotnine import ggplot, geom_line, aes, facet_grid, theme, element_text, geom_ribbon
from plotnine.scales import scale_y_log10, scale_x_log10
from plotnine.guides import guide_legend, guides

from scipy.stats import tukey_hsd

SOS = Sym('<SOS>')
EOS = Sym('<EOS>')

def get_pos_sequences(file):
    """Get POS tag sequences from a conllu treebank."""
    seqs = []
    with open(file, 'r') as f:
        for sent in tqdm(conllu.parse_incr(f)):
            # filter out multi-word tokens, then collect pos tags
            pos = [(None, SOS)] + [(Sym(token['upos']), Sym(token['upos'])) for token in sent if type(token['id']) is int] + [(EOS, EOS)]
            seqs.append(tuple(pos))
    return seqs

def estimate_conllu(file: str, sampling: list[int], fsas: int = 10):
    print(file)
    language = file.split('/')[1][:-7].capitalize()
    seqs = get_pos_sequences(file)
    fsa, samples, delta, tot = fsa_from_samples(seqs)
    true = estimate_entropy(fsa, samples, delta, tot, baseline=True)
    for estimator, val in true.items():
        print(f'{estimator:<30}: {val:>7.4f} nats ({(val - true["sMLE (pathsum)"])**2} MSE)')
    true = true['sMLE (pathsum)']

    # run sampling for various # of samples
    res = []

    # estimate across f FSAs
    for f in tqdm(range(fsas)):
        samples = []
        for num in sampling:
            random.shuffle(seqs)
            # samples += get_samples(fsa, num - len(samples))
            fsa_new, samples_new, delta_new, tot_new = fsa_from_samples(seqs[:num])
            # print(len(samples))
            # print(f'{"True":<30}: {true:>7.4f} nats')
            for estimator, val in estimate_entropy(fsa_new, samples_new, delta_new, tot_new, more=False).items():
                # print(f'{estimator:<30}: {val:>7.4f} nats')
                res.append({
                    'samples': num,
                    'method': estimator,
                    'entropy': val,
                    'MSE': (val - true)**2,
                    'MAB': val,
                    'lang': language
                })
    
    return res


def graph_convergence(tukey=False, graph=False):

    if tukey:
        results = []
        for file in glob.glob('data/*.conllu'):
            results += estimate_conllu(file, [10, 100, 1000], 0)

        grouped = defaultdict(lambda: defaultdict(list))
        for res in results:
            grouped[(res['lang'], res['samples'])][res['method']].append(res)

        # run and print test results
        with open('logs/ud.txt', 'w') as fout:
            for group in grouped:
                arrs = [np.array([y['MSE'] for y in x]) for x in grouped[group].values()]
                test = tukey_hsd(*arrs)
                fout.write('===========================\n')
                fout.write(f"{group}\n")
                fout.write(f"{list(grouped[group].keys())}\n")
                for arr in arrs:
                    fout.write(f"{arr.mean()} {arr.std()}\n")
                fout.write(f"{test}")

    # plot
    if graph:
        X = []
        for t in range(3): X.extend(list(range(2 * 10**t, 11 * 10**t, max(1, 10**t))))

        res = []
        for file in glob.glob('data/*.conllu'):
            res += estimate_conllu(file, X, 50)
        df = pd.DataFrame(res)

        # group by #samples, method, and language
        # calculate mean and standard error of MSE
        keep = ["samples", "method", "lang"]
        df = df.groupby(keep).agg(
            MAB=("MAB", "mean"),
            MAB_se=("MAB", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
            MSE=("MSE", "mean"),
            MSE_se=("MSE", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
        )

        df["MAB_lower"] = df["MAB"] - 1.96 * df["MAB_se"]
        df["MAB_upper"] = df["MAB"] + 1.96 * df["MAB_se"]
        df["MSE_lower"] = df["MSE"] - 1.96 * df["MSE_se"]
        df["MSE_upper"] = df["MSE"] + 1.96 * df["MSE_se"]
        df = df.reset_index()

        df = df.rename(columns={"MSE": "MSE_mean", "MAB": "MAB_mean"})
        df.drop(columns=["MSE_se", "MAB_se"], inplace=True)

        # Reshape DataFrame to long format using 'pd.wide_to_long()'
        df_long = pd.wide_to_long(df, stubnames=['MSE', 'MAB'], i=keep, j='bound', sep='_', suffix=r'(lower|upper|mean)').stack().reset_index()
        df_long = df_long.rename(columns={0: 'score', 'level_4': 'type'})

        # Pivot the DataFrame to reshape it
        df = df_long.pivot_table(index=keep + ['type'], 
                          columns='bound', values='score').reset_index()
        df.rename(columns={'type': 'metric', 'samples': '$|\\mathcal{D}|$'}, inplace=True)
        print(df)

        plot = (ggplot(df, aes(x='$|\\mathcal{D}|$', y='mean', ymin='lower', ymax='upper'))
            + geom_line(aes(color='method'))
            + geom_ribbon(aes(fill='method'), alpha=0.2)
            + facet_grid('metric~lang', scales='free_y')
            + scale_y_log10()
            + scale_x_log10()
            + theme(
                legend_title=element_text(size=0, alpha=0),
                axis_text_x=element_text(rotation=45),
                axis_title_y=element_text(size=0, alpha=0),
                legend_position="top",
                text=element_text(family='Times New Roman'),
            ))
        plot.draw(show=True)
        plot.save(filename='plots/ud.pdf', height=2.5, width=5)

def calc(file):
    seqs = get_pos_sequences(file)
    for estimator, val in estimate_entropy(*fsa_from_samples(seqs), more=True).items():
        print(f'{estimator:<30}: {val:>7.4f} nats')

def stats(file):
    seq = get_pos_sequences(file)
    fsa, samples, delta, tot = fsa_from_samples(seq)
    arcs = sum([len(x) for x in delta.values()])
    print(f'{file:<30}: {len(delta)}, {arcs}')

def main():
    assert len(sys.argv) == 2, "Usage: python3 ud.py <graph|tukey|stats>"
    if sys.argv[1] == 'graph':
        graph_convergence(graph=True)
    elif sys.argv[1] == 'tukey':
        graph_convergence(tukey=True)
    elif sys.argv[1] == 'stats':
        for file in glob.glob('data/*.conllu'):
            stats(file)
    else:
        raise ValueError("Usage: python3 ud.py <graph|stats>")

if __name__ == '__main__':
    main()