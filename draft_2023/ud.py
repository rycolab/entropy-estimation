"""
Experiment on structured entropy prediction using Markov model of POS tags from UD.
"""

from os import stat
import glob
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA

from utils import lift, fsa_from_samples, estimate_entropy, get_samples

import conllu
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
from collections import defaultdict

from plotnine import ggplot, geom_line, geom_point, aes, stat_smooth, facet_wrap, theme, element_text
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
    true = estimate_entropy(fsa, samples, delta, tot)['sMLE']

    # run sampling for various # of samples
    res = []

    # estimate across f FSAs
    for f in tqdm(range(fsas)):
        samples = []
        for num in sampling:
            random.shuffle(seqs)
            fsa, samples, delta, tot = fsa_from_samples(seqs[:num])
            samples += get_samples(fsa, num - len(samples))
            # print(f'{"True":<30}: {true:>7.4f} nats')
            for estimator, val in estimate_entropy(fsa, samples, delta, tot, more=False).items():
                # print(f'{estimator:<30}: {val:>7.4f} nats')
                res.append({
                    'samples': num,
                    'method': estimator,
                    'entropy': val,
                    'mse': (val - true)**2,
                    'lang': language
                })
    
    return res


def graph_convergence(tukey=False, graph=False):

    if tukey:
        results = []
        for file in glob.glob('data/*.conllu'):
            results += estimate_conllu(file, [10, 100, 1000], 100)

        grouped = defaultdict(lambda: defaultdict(list))
        for res in results:
            grouped[(res['lang'], res['samples'])][res['method']].append(res)
        for group in grouped:
            arrs = [np.array([y['mse'] for y in x]) for x in grouped[group].values()]
            test = tukey_hsd(*arrs)
            print('===========================')
            print(group)
            print(list(grouped[group].keys()))
            for arr in arrs:
                print(arr.mean(), arr.std())
            print(test)

    # plot
    if graph:
        X = []
        for t in range(3): X.extend(list(range(2 * 10**t, 11 * 10**t, max(1, 10**t))))

        res = []
        for file in glob.glob('data/*.conllu'):
            res += estimate_conllu(file, X, 10)

        df = pd.DataFrame(res)
        plot = (ggplot(df, aes(x='samples', y='mse', color='method',))
            + geom_line(stat='summary')
            + facet_wrap('~lang', nrow=2, ncol=3)
            + scale_y_log10()
            + scale_x_log10()
            + theme(legend_title=element_text(size=0, alpha=0),
                axis_text_x=element_text(rotation=45), legend_position=(0.8, 0.2))
            + guides(color=guide_legend(ncol=1)))
        plot.draw(show=True)
        plot.save(filename='plots/ud.pdf', height=3, width=4)

def calc(file):
    seqs = get_pos_sequences(file)
    for estimator, val in estimate_entropy(*fsa_from_samples(seqs), more=True).items():
        print(f'{estimator:<30}: {val:>7.4f} nats')

def main():
    graph_convergence(tukey=True)

if __name__ == '__main__':
    main()