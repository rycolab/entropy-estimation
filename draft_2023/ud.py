"""
Experiment on structured entropy prediction using Markov model of POS tags from UD.
"""

from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
from rayuela.fsa.sampler import Sampler

import entropy 

import conllu
from tqdm import tqdm
from collections import defaultdict

START = '<SOS>'
TERMINAL = '<EOS>'

def get_pos_transitions(file):
    """Get observed Markov transitions between POS tags from a conllu treebank."""
    
    transitions = defaultdict(lambda: defaultdict(int))
    with open(file, 'r') as f:
        for sent in tqdm(conllu.parse_incr(f)):
            # filter out multi-word tokens, then collect pos tags
            pos = [START] + [token['upos'] for token in sent if type(token['id']) is int] + [TERMINAL]
            # collect transitions
            for i in range(1, len(pos)):
                transitions[pos[i - 1]][pos[i]] += 1
    return transitions

def construct_fsa(transitions):
    """Construct the POS-tag WFSA given transition counts"""

    # map our states to ints
    conv = {TERMINAL: 1}
    for state in transitions:
        if state not in conv:
            conv[state] = len(conv) + 1
    
    # build actual WFSA
    fsa = FSA(Real)
    for s in transitions:
        tot = sum([transitions[s][t] for t in transitions[s]])
        for t in transitions[s]:
            fsa.add_arc(State(conv[s], s), Sym(t), State(conv[t], t), Real(transitions[s][t] / tot))
    
    # start/terminal nodes
    fsa.add_I(State(conv[START], START), Real(1.0))
    fsa.add_F(State(conv[TERMINAL], TERMINAL), Real(1.0))
    
    return fsa

def monte_carlo(fsa: FSA, it=1000):
    """Calculate MLE entropy using Monte-Carlo sampling"""

    sampler = Sampler(fsa)
    sample = sampler.ancestral(it)

    S, N, counts = entropy.prob(sample)
    singletons = len([x for x in counts if counts[x] == 1])
    mle = entropy.mle(S, N, counts)

    print(f'{mle:.2f} nats')
    print(f'{singletons} singletons out of {it}')
    return mle

def main():
    pos = get_pos_transitions('data/es_ancora-ud-train.conllu')
    fsa = construct_fsa(pos)
    monte_carlo(fsa)

if __name__ == '__main__':
    main()