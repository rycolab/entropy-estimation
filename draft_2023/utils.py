from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational, Semiring, expectation_semiring_builder
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
from rayuela.fsa.sampler import Sampler
from rayuela.fsa.pathsum import Pathsum, Strategy
from tqdm import tqdm

from collections import defaultdict
import entropy
import math
import numpy as np
from copy import deepcopy

# plotting settings (for latex)
import matplotlib.pyplot as plt
plt.rcParams.update({
    # "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

# make the NSB estimator shut up
import logging
logger = logging.getLogger()
logger.disabled = True

def sample(pfsa: FSA, sampler: Sampler, memo={}, return_memo=False):
    """Sample a path from the PFSA, including symbols on the arcs (for trajectory entropy)."""
    cur = sampler._draw({p : w for p, w in pfsa.I})
    output = [(None, cur)]

    while cur:
        if cur not in memo:
            D = {(a, j) : w for a, j, w in pfsa.arcs(cur)}
            D[(0, 0)] = pfsa.ρ[cur]
            vec, store = np.zeros((len(D))), {}
            for p, w in D.items(): 
                vec[len(store)] = float(w)
                store[len(store)] = p
            vec /= vec.sum()
            memo[cur] = (vec, store)
        
        (a, cur) = memo[cur][1][np.random.choice(len(memo[cur][1]), p=memo[cur][0])]
        if a != 0: output.append((a, cur))

    if return_memo:
        return tuple(output), memo
    return tuple(output)

def get_samples(fsa: FSA, samples: int, prog=False):
    sampler = Sampler(fsa)
    memo = {}
    s = []
    for _ in range(samples) if not prog else tqdm(range(samples)):
        res, memo = sample(fsa, sampler, memo, return_memo=True)
        s.append(res)
    del memo
    return s

def lift(old_fsa: FSA, func):
    """Lift an FSA to the expectation semiring, from p -> <p, val(p)>
    - old_fsa: the original FSA, whose copy will be returned
    - func: arbitrary function for which we want the expectation over paths"""
    # init semiring
    exp = expectation_semiring_builder(Real, Real)

    # lift our old fsa onto the expectation semiring
    fsa = FSA(exp)
    s, e = None, None
    for p, w in old_fsa.I:
        s = p
        fsa.set_I(p, w=exp.one)
    for p, w in old_fsa.F:
        e = p
        fsa.set_F(p, w=exp.one)
    for p in old_fsa.Q:
        for a, q, w in old_fsa.arcs(p):
            l = func(w)
            fsa.add_arc(p, a, q, w=exp(l[0], l[1]))
    return fsa

def fsa_from_samples(samples: list[tuple]):
    """Generate MLE structure of FSA based on sampling x values
    - samples: a list of trajectories from which the transitions in the PFSA will be created
    """
    delta = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)

    # construct new transition function, add dummy nodes to start and end
    for samp in samples:
        last = None
        tot[None] += 1
        for (symbol, state) in samp:
            delta[last][(symbol, state)] += 1
            tot[state] += 1
            last = state
        delta[last][(None, None)] += 1
    
    # make FSA
    fsa = FSA(Real)
    for i in delta:
        for (symbol, j) in delta[i]:
            # initials and finals (dummy Nones)
            if j is None: fsa.set_F(State(i), Real(1.0))
            elif i is None: fsa.set_I(State(j), Real(delta[i][(symbol, j)] / tot[i]))
            # actual transitions
            else: fsa.add_arc(State(i), Sym(symbol), State(j), Real(delta[i][(symbol, j)] / tot[i]))

    return fsa, samples, delta, tot

def exact_entropy(fsa: FSA):
    """Calculate the exact entropy of an FSA, using the pathsum algorithm"""
    # lift to expectation semiring
    entropy_fsa = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))
    return float(entropy_fsa.pathsum().score[1])

def estimate_entropy(fsa: FSA, samples, delta, ct, more=False, less=False, baseline=False):
    """Calculate the entropy estimates, given the samples and the correspondingly constructed MLE FSA
    - fsa: the FSA whose entropy is being estimated
    - samples: the samples from which the estimation is done
    - delta: transition function
    - more: whether to use only NSB or other estimators too
    - baseline: whether to calculate sMLE with pathsum or use decomposed method
    """
    res = defaultdict(float)

    # lift to expectation semiring
    if baseline:
        entropy_fsa = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))
        res['sMLE (pathsum)'] = float(entropy_fsa.pathsum().score[1])
        res['sMLE (state-elim)'] = float(exp_semiring(fsa)[1].score[1])

    # simple estimates to get
    counts = entropy.prob(samples)
    res['uMLE'] = entropy.mle(*counts)
    if not less:
        try:
            res['uNSB'] = entropy.nsb(*counts)
        except:
            res['uNSB'] = res['uMLE']

    # structured NSB (or other) estimator
    if not less:
        for state in ct:
            N = sum(delta[state].values())
            ct_q = ct[state] / len(samples)
            dist_q = [x / N for x in delta[state].values()], N, delta[state]
            diff = ct_q * entropy.mle(*dist_q)
            if more:
                for func in entropy.funcs:
                    if func == entropy.mle:
                        res['sMLE'] += diff
                        continue
                    try:
                        res[f's{str(func.__name__).capitalize()}'] += ct_q * func(*dist_q)
                    except:
                        res[f's{str(func.__name__).capitalize()}'] += diff
            else:
                res['sMLE'] += diff
                try:
                    res['sNSB'] += ct_q * entropy.nsb(*dist_q)
                except:
                    res['sNSB'] += diff

    
    return res

def state_elim_pathsum(fsa: FSA):
    print(len(fsa.Q))

    # get start, end
    s, e = [], []
    for p, w in fsa.I:
        s.append(p)
    for p, w in fsa.F:
        e.append(p)
    ig = s + e

    state2symbol = {}
    for q in fsa.Q:
        for sym in fsa.δ[q]:
            corresp = list(fsa.δ[q][sym].keys())[0]
            state2symbol[corresp] = sym
    # eliminate states
    d = set()
    elim = set()
    for q in fsa.Q:
        if q in ig or q not in state2symbol: continue
        loop = fsa.δ[q][state2symbol[q]][q]

        elim.add(q)

        for pr in fsa.Q:
            if pr in elim: continue
            incoming = fsa.δ[pr][state2symbol[q]][q]

            for su in fsa.Q:
                if su in elim or su not in state2symbol: continue
                outgoing = fsa.δ[q][state2symbol[su]][su]
                comb = incoming * loop.star() * outgoing
                fsa.δ[pr][state2symbol[su]][su] += comb

    return fsa.δ[s[0]][state2symbol[e[0]]][e[0]]


def exp_semiring(old_fsa: FSA):

    # lift to expectation semiring
    fsa = lift(old_fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))

    # print entropy
    true = state_elim_pathsum(fsa)
    print(true)
    
    return fsa, true