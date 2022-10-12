"""
Experiment on structured entropy prediction using Markov model of POS tags from UD.
"""

import math
from os import stat
from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational, Semiring, expectation_semiring_builder
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
from rayuela.fsa.sampler import Sampler
from rayuela.fsa.pathsum import Pathsum, Strategy

import entropy 
from utils import lift, fsa_from_samples, estimate_entropy, get_samples

import matplotlib.pyplot as plt

import conllu
from tqdm import tqdm
from collections import defaultdict
import random

START = '<SOS>'
TERMINAL = '<EOS>'

def get_pos_sequences(file):
    """Get POS tag sequences from a conllu treebank."""
    seqs = []
    with open(file, 'r') as f:
        for sent in tqdm(conllu.parse_incr(f)):
            # filter out multi-word tokens, then collect pos tags
            pos = [token['upos'] for token in sent if type(token['id']) is int] + [TERMINAL]
            seqs.append(tuple(pos))
    return seqs

def graph_convergence(file):
    seqs = get_pos_sequences(file)
    true = estimate_entropy(*fsa_from_samples(seqs))['Structured MLE (pathsum)']

    # run sampling for various # of samples
    X = list(range(1, 100, 1))
    Ys = defaultdict(list)

    # estimate
    for num in tqdm(X):
        random.shuffle(seqs)
        fsa, samples, delta, tot = fsa_from_samples(seqs[:num])
        samples = get_samples(fsa, num)
        for estimator, val in estimate_entropy(fsa, samples, delta, tot, more=False).items():
            print(f'{estimator:<30}: {val:>7.4f} nats')
            Ys[estimator].append(val)
    
    # plot
    for Y in Ys:
        plt.plot(X, Ys[Y], label=Y)
    plt.axhline(y=true, color='r', label='True')
    plt.xlabel('# Sentences')
    plt.ylabel('Entropy (nats)')
    plt.title(f'# Samples vs. Entropy')
    plt.legend()
    plt.show()

def calc(file):
    seqs = get_pos_sequences(file)
    for estimator, val in estimate_entropy(*fsa_from_samples(seqs), more=True).items():
        print(f'{estimator:<30}: {val:>7.4f} nats')

def main():
    graph_convergence('data/es_ancora-ud-train.conllu')

if __name__ == '__main__':
    main()