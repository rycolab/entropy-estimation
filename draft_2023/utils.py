from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational, Semiring, expectation_semiring_builder
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
from rayuela.fsa.sampler import Sampler
from rayuela.fsa.pathsum import Pathsum, Strategy


def lift(old_fsa: FSA, func):
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