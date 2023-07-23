from collections import defaultdict
import math
import sys

from rayuela.base.semiring import Real
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State

from scipy.special import digamma, polygamma
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import lift, fsa_from_samples, estimate_entropy, get_samples

from plotnine import (
    ggplot,
    geom_line,
    geom_ribbon,
    aes,
    theme,
    element_text,
    facet_wrap,
    facet_grid,
    geom_histogram
)
from plotnine.scales import scale_y_log10, scale_x_log10

from scipy.stats import tukey_hsd

def make_acyclic_machine(states=3):
    """Make an acyclic FSA (homomorphic to a DAG) with outgoing weights from a node summing to 1"""
    fsa = FSA(Real)
    for i in range(states):
        # use Dirichlet to generate outgoing weights
        s = np.random.dirichlet([1.0] * (states - i - 1), 1).tolist()[0]
        for w, j in zip(s, range(i + 1, states)):
            fsa.add_arc(State(i), Sym(j), State(j), Real(w))

    fsa.set_I(State(0), Real(1.0))
    fsa.set_F(State(states - 1), Real(1.0))

    return fsa


def make_cyclic_machine(states=3):
    """Make a cyclic FSA (homomorphic to a complete directed graph) with outgoing weights from a node summing to 1"""
    fsa = FSA(Real)
    for i in range(states - 1):
        # use Dirichlet to generate outgoing weights
        s = np.random.dirichlet([1.0] * states, 1).tolist()[0]
        for j in range(states):
            fsa.add_arc(State(i), Sym(j), State(j), Real(s[j]))

    fsa.set_I(State(0), Real(1.0))
    fsa.set_F(State(states - 1), Real(1.0))

    return fsa


def run_iter(fsa: FSA, samples=None, num_samps=1000, more=False):
    """Estimate structured and unstructured entropy given a true FSA and number of samples to get"""
    if not samples:
        samples = get_samples(fsa, num_samps)
    return estimate_entropy(*fsa_from_samples(samples), more=more)

def measure_convergence(states, sampling, cyclic, resample, fsas, prog=True):
    """Measure convergence of entropy estimates to true value over sample size for a single FSA"""
    # run sampling for various # of samples
    X = sampling
    results = []

    for f in tqdm(range(fsas)):
        # make FSA and get true entropy
        fsa = (
            make_cyclic_machine(states=states)
            if cyclic
            else make_acyclic_machine(states=states)
        )
        lifted = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))
        true = float(lifted.pathsum().score[1])

        # if resample is True, then we generate a new sample every time, otherwise we keep one throughout
        s = None if resample else get_samples(fsa, X[-1])

        for samp in tqdm(X) if prog else X:
            res = run_iter(fsa, samples=s[:samp] if s else None, num_samps=samp)
            for key in res:
                results.append(
                    {
                        "samples": samp,
                        "method": key,
                        "MSE": ((res[key] - true) ** 2),
                        "MAB": (abs(res[key] - true)),
                        "cyclic": cyclic,
                        "states": states,
                    }
                )

    return results


def graph_convergence(
    states: list[int],
    sampling: list[int],
    cyclic: bool,
    resample: bool,
    fsas: int,
    tukey: bool = False,
    graph: bool = False,
):
    # get results
    results = []
    sampling = sampling
    for state in states:
        results.extend(
            measure_convergence(state, sampling, cyclic, resample, fsas, prog=False)
        )

    # significance test
    if tukey:
        grouped = defaultdict(lambda: defaultdict(list))
        for res in results:
            grouped[(res["states"], res["cyclic"], res["samples"])][
                res["method"]
            ].append(res)
        for group in grouped:
            arrs = [np.array([y["MSE"] for y in x]) for x in grouped[group].values()]
            test = tukey_hsd(*arrs)
            print("===========================")
            print(group)
            print(list(grouped[group].keys()))
            for arr in arrs:
                print(arr.mean(), arr.std())
            print(test)

    # make graph
    if graph:
        df = pd.DataFrame(results)

        # group by #samples, method, and language
        # calculate mean and standard error of MSE
        keep = ["samples", "method", "states"]
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
        df.rename(columns={'states': '$|Q|$', 'type': 'metric', 'samples': '$|\\mathcal{D}|$'}, inplace=True)
        print(df)

        plot = (
            ggplot(df, aes(x="$|\\mathcal{D}|$", y="mean", ymin="lower", ymax="upper"))
            + geom_line(aes(color="method"))
            + geom_ribbon(aes(fill="method"), alpha=0.2)
            + facet_grid("metric~$|Q|$", scales="free")
            + scale_y_log10()
            + scale_x_log10()
            + theme(
                legend_title=element_text(size=0, alpha=0),
                axis_text_x=element_text(rotation=45),
                axis_title_y=element_text(size=0, alpha=0),
                legend_position="top",
                text=element_text(family='Times New Roman'),
            )
        )
        plot.draw(show=True)
        plot.save(
            filename=f'plots/synthetic-{states}-{"cyclic" if cyclic else "acyclic"}-all.pdf',
            height=2.5,
            width=5,
        )

def diff_distrib(states: int=10):
    diffs = []
    for i in tqdm(range(1000)):
        fsa = make_acyclic_machine(states=states)
        lifted = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))
        true = float(lifted.pathsum().score[1])
        res = run_iter(fsa, num_samps=10)
        diff = (res["sMLE"] - true)**2 - (res["uMLE"] - true)**2
        diffs.append(diff)
    
    # plot histogram of differences
    df = pd.DataFrame(diffs, columns=["diff"])
    plot = (
        ggplot(df, aes(x="diff"))
        + geom_histogram(bins=20)
        + theme(
            legend_title=element_text(size=0, alpha=0),
            axis_text_x=element_text(rotation=45),
            axis_title_y=element_text(size=0, alpha=0),
            legend_position="top",
            text=element_text(family='Times New Roman'),
        )
    )
    plot.draw(show=True)

def var_ent(K: int=2):
    """Variance of entropy for a uniform Dirichlet distribution with K classes, per Archer (2014)"""
    A = K
    res = 0.0
    if A != 0:
        res += K * (2 / ((A + 1) * A)) * ((digamma(3) - digamma(A + 2))**2 + polygamma(1, 3) - polygamma(1, A + 2))
        res += (((K * (K - 1)) / 2 - 1) / ((A + 1) * A)) * ((digamma(2) - digamma(A + 2)) * (digamma(2) - digamma(A + 2)) - polygamma(1, A + 2))
    return res

def main():
    assert len(sys.argv) == 3, "Usage: python synthetic.py <tukey|graph> <cylic|acyclic>"

    cyclic = False

    # set cyclic
    if sys.argv[2] == "cyclic":
        cyclic = True
    elif sys.argv[2] == "acyclic":
        cyclic = False
    else:
        raise ValueError("Usage: python synthetic.py <tukey|graph> <cylic|acyclic>")

    # run tests
    if sys.argv[1] == "tukey":
        graph_convergence(
            states=[5, 10, 20, 50],
            sampling=[10, 100],
            cyclic=cyclic,
            resample=True,
            fsas=20,
            tukey=True,
        )
    elif sys.argv[1] == "graph":
        X = []
        for t in range(4):
            X.extend(list(range(2 * 10**t, 11 * 10**t, max(1, 10**t))))
        graph_convergence(
            states=[5, 10, 20, 50],
            sampling=X,
            cyclic=cyclic,
            resample=False,
            fsas=20,
            graph=True,
        )
    else:
        raise ValueError("Usage: python synthetic.py <tukey|graph> <cylic|acyclic>")

if __name__ == "__main__":
    for i in range(100):
        print(i, var_ent(i))
