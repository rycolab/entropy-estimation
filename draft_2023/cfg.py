import glob
import entropy
import pandas as pd
import numpy as np
from collections import defaultdict
import sys

from plotnine import ggplot, geom_line, aes, facet_wrap, theme, element_text, geom_ribbon
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

def _sample_cfg(cfg: CFG, keep_str=True, rhs=None, weight=None):
    """Sample a leftmost derivation from the cfg."""
    sample = [(None, [cfg.S])]
    buffer = [cfg.S]
    done = []

    if not rhs:
        # precompute transitions
        rhs = defaultdict(list)
        weight = defaultdict(list)
        for (head, body), w in cfg.P:
            rhs[head].append(body)
            weight[head].append(w)
    rands = defaultdict(list)

    # terminate if all are terminals
    # trying to optimise this: start only at the leftmost non-terminal each time (pos)
    pos = 0
    while buffer:
        # pop terminals
        if buffer[-1] not in cfg.V:
            done.append(buffer.pop())
            continue

        # expand leftmost nonterminal
        rule = None
        sym = buffer.pop()

        # choose per probability
        if len(rands[sym]) == 0:
            rands[sym] = np.random.choice(len(weight[sym]), size=100, p=weight[sym]).tolist()
        index = rands[sym].pop()
        for i in range(len(rhs[sym][index]) - 1, -1, -1):
            buffer.append(rhs[sym][index][i])

        rule = Production(sym, rhs[sym][index])
        sample.append((rule, None))
    
    # keep last str
    sample[-1] = (sample[-1][0], done)
    return sample

def sample_cfg(cfg: CFG, n: int, keep_str=True):
    """Sample n leftmost derivations from the cfg."""

    # precompute transitions
    rhs = defaultdict(list)
    weight = defaultdict(list)
    for (head, body), w in cfg.P:
        rhs[head].append(body)
        weight[head].append(w)
    
    # sample
    samples = [_sample_cfg(cfg, keep_str, rhs, weight) for _ in range(n)]
    return samples

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
    probs = entropy.prob(strs)
    res['uMLE'] = entropy.mle(*probs)
    try:
        res['uNSB'] = entropy.nsb(*probs)
    except:
        res['uNSB'] = res['uMLE']

    # structured NSB (or other) estimator
    for head in delta:
        N = ct[head]
        ct_q = ct[head] / len(samples)
        dist_q = [x / N for x in delta[head].values()], N, list(delta[head].values())
        if more:
            for func in entropy.funcs:
                res[f'Structured {func.__name__}'] += ct_q * func(*dist_q)
        else:
            diff = ct_q * entropy.mle(*dist_q)
            res['sMLE'] += diff
            try:
                res['sNSB'] += ct_q * entropy.nsb(*dist_q)
            except:
                res['sNSB'] += diff
    
    return res

def graph_convergence():
    """Graph estimator convergence"""
    X = []
    for t in range(5): X.extend(list(range(2 * 10**t, 11 * 10**t, max(1, 10**t))))
    res = []

    # estimate cfg entropies
    cfgs = [(load_cfg(file), file.split('/')[-1].split('-')[0]) for file in glob.glob("data/pcfg/*")]
    # cfgs = simple_cfgs()
    for cfg, name in cfgs:
        for run in tqdm(range(20)):
            s = sample_cfg(cfg, keep_str=False, n=X[-1])
            for num in X:
                inputs = cfg_from_samples(s[:num])
                for estimator, val in estimate_entropy(*inputs).items():
                    # print(f'{estimator:<30}: {val:>7.4f} nats')
                    res.append({
                        'samples': num,
                        'method': estimator,
                        'entropy': val,
                        'cfg': name
                    })

    df = pd.DataFrame(res)

    # group by #samples, method, and language
    # calculate mean and standard error of mse
    df = df.groupby(['samples', 'method']).agg(
        entropy=('entropy', 'mean'),
        entropy_se=('entropy', lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))))
    )

    df['entropy_lower'] = df['entropy'] - 1.96 * df['entropy_se']
    df['entropy_upper'] = df['entropy'] + 1.96 * df['entropy_se']
    df = df.reset_index()
    df = df.rename(columns={'entropy': 'Entropy estimate (nats)', 'samples': '$|\\mathcal{D}|$', 'method': 'Method'})

    plot = (ggplot(df, aes(x='$|\\mathcal{D}|$', y='Entropy estimate (nats)', ymin='entropy_lower', ymax='entropy_upper'))
        + geom_line(aes(color='Method'))
        + geom_ribbon(aes(fill='Method'), alpha=0.2)
        + scale_x_log10()
        + theme(
            legend_title=element_text(size=0, alpha=0),
            axis_text_x=element_text(rotation=45),
            text=element_text(family='Times'),
        )
    )
    
    plot.draw(show=True)
    plot.save(filename='plots/cfg.pdf', height=2, width=3)

def stats():
    cfgs: list[tuple[CFG, str]] = [(load_cfg(file), file.split('/')[-1].split('-')[0]) for file in glob.glob("data/pcfg/*")]
    for cfg, name in cfgs:
        print(f"Alphabet: {len(cfg.Sigma)}")
        print(f"Nonterminals: {len(cfg.V)}")
        print(f"Productions: {len(cfg._P)}")

def main():
    assert len(sys.argv) == 2, "Usage: python3 cfg.py <graph|stats>"
    if sys.argv[1] == 'graph':
        graph_convergence()
    elif sys.argv[1] == 'stats':
        stats()
    else:
        raise ValueError("Usage: python3 cfg.py <graph|stats>")

if __name__ == "__main__":
    main()