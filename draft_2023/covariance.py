from utils import estimate_entropy, get_samples, fsa_from_samples, exact_entropy
from plotnine import ggplot, geom_point, aes, geom_histogram, geom_vline, facet_wrap
import pandas as pd
from tqdm import tqdm

from rayuela.base.semiring import Real
from rayuela.fsa.fsa import FSA
import random

# p
def run(p=0.5, bucket=100, size=10000):
    # make PFSA with one self-looping state
    fsa = FSA(Real)
    fsa.add_state(0)
    fsa.set_I(0, Real(1.0))

    states = 100
    fsa.add_state(states)
    fsa.set_F(states, Real(1.0))

    for i in range(1, states):
        fsa.add_state(i)
        fsa.add_arc(0, str(i), i, Real(1 / (states - 1)))
        chance = random.random()
        fsa.add_arc(i, str(i), i, Real(chance))
        fsa.add_arc(i, str(states), states, Real(1 - chance))

    # get samples
    samples = get_samples(fsa, size, prog=True)
    ent = estimate_entropy(*fsa_from_samples(samples), more=True, baseline=True)
    true = ent['sMLE (pathsum)']
    for key, val in ent.items():
        print(f'{key:<20}: {val:>10.7f} nats ({(val - true)**2:.7f} MSE)')
    
    random.shuffle(samples)

    # estimate entropy at state 1 over sets of 10
    obs = []
    for i in tqdm(range(0, size, bucket)):
        ent = estimate_entropy(*fsa_from_samples(samples[i:i + bucket]), more=True)
        count = sum([len(s) - 2 for s in samples[i:i + bucket]]) / bucket
        for key in ent:
            obs.append({
                'count': count,
                'entropy': ent[key],
                'estimator': key
            })

    # make df
    df = pd.DataFrame(obs)

    # plot
    # plot = ggplot(df, aes(x='count', y='entropy')) + geom_point(alpha=0.1)
    # print(plot)
    # plot.save(f'plots/covariance-{p}.png')

    plot = (ggplot(df, aes(x='entropy')) + geom_histogram(alpha=0.3)
            + geom_vline(xintercept=true) + facet_wrap('~estimator'))
    print(plot)

run(0.5)