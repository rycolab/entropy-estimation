import glob
import entropy
from collections import defaultdict
import numpy as np

from rayuela.cfg.cfg import CFG, Production
from rayuela.base.semiring import Real
from tqdm import tqdm

# make the NSB estimator shut up
import logging
logger = logging.getLogger()
logger.disabled = True

def load_cfg(file: str):
    # load
    cfg = CFG.get_moore_cfg(file, R=Real)
    
    # normalise (make probabilistic)
    tot = defaultdict(lambda: Real(0.0))
    for p, w in cfg.P:
        (head, body) = p
        tot[head] += w
    for prodrule in cfg._P:
        cfg._P[prodrule] /= tot[prodrule[0]]

    return cfg

def sample_cfg(cfg: CFG, keep_str=True):
    """Sample a leftmost derivation from the cfg."""
    sample = [(None, [cfg.S])]
    cur = [cfg.S]

    # precompute transitions
    rhs = defaultdict(list)
    weight = defaultdict(list)
    for (head, body), w in cfg.P:
        rhs[head].append(body)
        weight[head].append(w)

    # terminate if all are terminals
    while any([x in cfg.V for x in cur]):
        new_cur = []

        # expand leftmost nonterminal
        done = False
        rule = None
        for i, sym in enumerate(cur):
            if done:
                new_cur.append(sym)
            elif sym in cfg.V:
                # choose per probability
                index = np.random.choice(len(weight[sym]), p=weight[sym])
                for out in rhs[sym][index]:
                    new_cur.append(out)
                rule = Production(sym, rhs[sym][index])
                done = True
            else:
                new_cur.append(sym)
        
        sample.append((rule, new_cur if keep_str else None))
        cur = new_cur
    
    # keep last str
    sample[-1] = (sample[-1][0], new_cur)
    return sample

def cfg_from_samples(samples: list[tuple[Production, list]]):
    """Construct a CFG from some samples"""
    # make transition counts
    delta = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)

    # construct
    for sample in samples:
        for (head, body), _ in sample[1:]:
            delta[head][body] += 1
    
    # make cfg
    cfg = CFG()
    for head in delta:
        tot[head] = sum(delta[head].values())
        for body in delta[head]:
            cfg.add(delta[head][body] / tot[head], head, *body)
    
    return cfg, samples, delta, tot

def estimate_entropy(cfg: CFG, samples, delta, ct, more=False, baseline=True):
    """Estimate the entropy of a CFG"""
    # TODO: Lehmann style pathsum for cfg possible?
    res = defaultdict(float)

    # simple estimates to get
    strs = [tuple(s[-1][1]) for s in samples]
    res['uMLE'] = entropy.mle(*entropy.prob(strs))
    res['uNSB'] = entropy.nsb(*entropy.prob(strs))

    # structured NSB (or other) estimator
    for (head, body), w in cfg.P:
        N = sum(delta[head].values())
        ct_q = ct[head] / len(samples)
        dist_q = [x / N for x in delta[head].values()], N, delta[head]
        if more:
            for func in entropy.funcs:
                res[f'Structured {func.__name__}'] += ct_q * func(*dist_q)
        else:
            res['sNSB'] += ct_q * entropy.nsb(*dist_q)
            if not baseline:
                res['sMLE'] += ct_q * entropy.mle(*dist_q)
    
    return res

def graph_convergence(cfg: CFG):
    for i in [10, 100, 1000, 10000]:
        s = [sample_cfg(cfg, keep_str=False) for _ in tqdm(range(i))]
        print(estimate_entropy(*cfg_from_samples(s)))

def estimate_from_file(files: list[str]):
    for file in files:
        cfg = load_cfg(file)
        graph_convergence(cfg)

def main():
    estimate_from_file(list(glob.glob("data/pcfg/*")))

if __name__ == "__main__":
    main()