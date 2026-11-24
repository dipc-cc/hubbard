import pytest
import numpy as np
import sisl

from hubbard import HubbardHamiltonian, sp2, NEGF


def test_negf_bias_not_implemented():
    # Finite bias is not supported and must fail before any set-up
    He = sp2(sisl.geom.zgnr(2))
    E = HubbardHamiltonian(He, U=3., kT=0.025)
    HC = He.tile(3, axis=0)
    HC.set_nsc([1, 1, 1])
    idx = [range(len(He)), range(len(HC.H) - len(He), len(HC.H))]
    D = HubbardHamiltonian(HC.H, n=np.tile(E.n, 3), U=3., kT=0.025)
    He_k = E.H.Hk().toarray()
    with pytest.raises(NotImplementedError, match="bias"):
        NEGF(D, [(E, '-A'), (E, '+A')], idx, V=0.1)
    # The electrodes were not shifted to their chemical potentials
    assert np.array_equal(E.H.Hk().toarray(), He_k)
