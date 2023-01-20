import glob
import entropy
import pandas as pd
import numpy as np
from collections import defaultdict

from plotnine import ggplot, geom_line, aes, facet_wrap, theme, element_text
from plotnine.scales import scale_y_log10, scale_x_log10
from plotnine.guides import guide_legend, guides

from rayuela.cfg.cfg import CFG, Production
from rayuela.base.semiring import Real
from tqdm import tqdm

# make the NSB estimator shut up
import logging
logger = logging.getLogger()
logger.disabled = True

def simple_cfgs():
    """Make some simple cfgs with known entropy to test on"""
    cfgs = []

    # 0.69 nats
    # cfgs.append(CFG.from_string("""
    # S → A:1.0
    # A → a:0.5
    # A → b:0.5
    # """.strip(), Real))

    # cfgs.append(CFG.from_string("""
    # S → A:1.0
    # A → A:0.5
    # A → a:0.5
    # """.strip(), Real))

    # cfgs.append(CFG.from_string("""
    # S → B:1.0
    # B → A:1.0
    # A → A:0.5
    # A → a:0.5
    # """.strip(), Real))

    # cfgs.append(CFG.from_string("""
    # S → C:1.0
    # C → B:1.0
    # B → A:1.0
    # A → A:0.5
    # A → a:0.5
    # """.strip(), Real))

    return cfgs

def load_cfg(file: str):
    """Get Moore-type cfg from file and normalise it"""
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

def estimate_entropy(cfg: CFG, samples, delta, ct, more=False):
    """Estimate the entropy of a CFG"""
    # TODO: Lehmann style pathsum for cfg possible?
    res = defaultdict(float)

    # simple estimates to get
    strs = [tuple(s[0] for s in sample) for sample in samples]
    res['uMLE'] = entropy.mle(*entropy.prob(strs))
    res['uNSB'] = entropy.nsb(*entropy.prob(strs))

    # structured NSB (or other) estimator
    for head in delta:
        N = sum(delta[head].values())
        ct_q = ct[head] / len(samples)
        dist_q = [x / N for x in delta[head].values()], N, delta[head]
        if more:
            for func in entropy.funcs:
                res[f'Structured {func.__name__}'] += ct_q * func(*dist_q)
        else:
            res['sNSB'] += ct_q * entropy.nsb(*dist_q)
            res['sMLE'] += ct_q * entropy.mle(*dist_q)
    
    return res

def graph_convergence():
    """Graph estimator convergence"""
    X = []
    for t in range(4): X.extend(list(range(2 * 10**t, 11 * 10**t, max(1, 10**t))))
    res = []

    cfgs = [load_cfg(file) for file in glob.glob("data/pcfg/*")]
    # cfgs = simple_cfgs()
    for cfg in cfgs:
        s = []
        for num in tqdm(X):
            s.extend([sample_cfg(cfg, keep_str=False) for _ in range(num - len(s))])
            for estimator, val in estimate_entropy(*cfg_from_samples(s)).items():
                print(f'{estimator:<30}: {val:>7.4f} nats')
                res.append({
                    'samples': num,
                    'method': estimator,
                    'entropy': val
                })

    df = pd.DataFrame(res)
    plot = (ggplot(df, aes(x='samples', y='entropy', color='method',))
        + geom_line(stat='summary')
        # + facet_wrap('~lang', nrow=2, ncol=3)
        + scale_x_log10()
        + theme(axis_text_x=element_text(rotation=45)))
    plot.draw(show=True)
    plot.save(filename='plots/cfg.pdf', height=3, width=4)

def main():
    graph_convergence()

if __name__ == "__main__":
    main()