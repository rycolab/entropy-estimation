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
    res = []

    with open(file, 'r') as f:
        for sent in tqdm(conllu.parse_incr(f)):
            # filter out multi-word tokens, then collect pos tags
            pos = [START] + [token['upos'] for token in sent if type(token['id']) is int] + [TERMINAL]
            res.append(tuple(pos))
            # collect transitions
            for i in range(1, len(pos)):
                transitions[pos[i - 1]][pos[i]] += 1

    return transitions, res

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

def monte_carlo(fsa: FSA, it=100):
    """Calculate MLE entropy using Monte-Carlo sampling"""

    sampler = Sampler(fsa)
    sample = sampler.ancestral(it)

    S, N, counts = entropy.prob(sample)
    singletons = len([x for x in counts if counts[x] == 1])
    mle = entropy.mle(S, N, counts)

    print(f'MC: {mle:.3f} nats')
    print(f'{singletons} singletons out of {it}')
    return mle

def _state_nsb_iter(fsa: FSA, sampler: Sampler, samp=100):
    sample = sampler.ancestral(samp)

    # store raw data to pass up
    count = defaultdict(int)
    samples = defaultdict(list)

    for s in sample:

        # collect samples for prob q is in path
        for q in set(s):
            count[q] += 1
        
        # collect samples for transition prob distrib from q
        prev = '<SOS>'
        for q in s:
            samples[prev].append(q)
            prev = q
    
    # calculate sub-sample stats
    p_q = {x: count[x] / samp for x in count}
    H_q = {x: entropy.mle(*entropy.prob(samples[x])) for x in samples}

    return count, samples, p_q, H_q


def state_nsb(fsa: FSA, it=100, samp=100):

    sampler = Sampler(fsa)

    count_q = defaultdict(int)
    samples_q = defaultdict(list)
    subsamps = []

    # get subsamples to calculate covariance
    for _ in tqdm(range(it)):
        count, samples, p_q, H_q = _state_nsb_iter(fsa, sampler, samp)
        for q in count:
            count_q[q] += count[q]
            samples_q[q].extend(samples[q])
        subsamps.append((p_q, H_q))

    # global vals over all samples
    exp_p_q = {x: count_q[x] / (samp * it) for x in count_q}
    exp_H_q = {x: entropy.mle(*entropy.prob(samples_q[x])) for x in samples_q}
    exp_covar_q = {}
    
    # calculate covariances over subsamples
    for q in count_q:
        covars = []
        for subsamp in subsamps:
            p_q = subsamp[0].get(q, 0)
            H_q = subsamp[1].get(q, 0)
            covars.append((p_q - exp_p_q.get(q, 0)) * (H_q - exp_H_q.get(q, 0)))
        exp_covar = sum(covars) / len(covars)
        exp_covar_q[q] = exp_covar

    # print vals
    for q in count_q:
        print(f'{q}: p(q) = {exp_p_q[q]:.3f}, H_q = {exp_H_q.get(q, 0):.3f}, covar(p(q), H_q) = {exp_covar_q[q]:.5f}')
    
    # entropy estimate
    res = 0.0
    for q in count_q:
        res += exp_p_q[q] * exp_H_q[q] + exp_covar_q[q]

    print(f'Smart way: {res:.3f} nats')
    return res
            

def main():
    pos, seqs = get_pos_transitions('data/es_ancora-ud-train.conllu')
    fsa = construct_fsa(pos)

    ground_truth = entropy.mle(*entropy.prob(seqs))
    print(f'Ground truth: {ground_truth:.3f} nats')

    monte_carlo(fsa)
    state_nsb(fsa)

if __name__ == '__main__':
    main()