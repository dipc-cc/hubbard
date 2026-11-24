import pytest
import numpy as np

import hubbard.hamiltonian as hh
import hubbard.sp2 as sp2
import hubbard.density as density
import sisl


def test_quick():
    molecule = sisl.geom.agnr(7).tile(3, 0)
    molecule.set_nsc([1, 1, 1])
    Hsp2 = sp2(molecule)
    H = hh.HubbardHamiltonian(Hsp2, U=3.5)
    H.random_density()
    dn = H.iterate(density.calc_n_insulator, mixer=sisl.mixing.LinearMixer())
    H.write_density('test.nc')
    H.write_initspin('test.fdf')


@pytest.mark.parametrize("polarize", [
    lambda H: H.set_polarization([0], dn=[3]),
    lambda H: H.polarize_sublattices(),
], ids=["set_polarization", "polarize_sublattices"])
def test_polarization_unpolarized_raises(polarize):
    # Spin polarization is meaningless with a single spin channel
    H = hh.HubbardHamiltonian(sp2(sisl.geom.zgnr(2), spin='unpolarized'), U=3.)
    n0 = H.n.copy()
    with pytest.raises(ValueError, match="spin-polarized Hamiltonian"):
        polarize(H)
    assert np.array_equal(H.n, n0)


def _polarized_zgnr2():
    # Four sites with q = (3, 1): the initial n = 0.5 is normalized to
    # n_up = 0.75 and n_dn = 0.25 on every site
    H = hh.HubbardHamiltonian(sp2(sisl.geom.zgnr(2)), U=3., q=(3., 1.))
    assert np.allclose(H.n, [[0.75] * 4, [0.25] * 4])
    return H


def test_set_polarization_polarized():
    # Up sites are set to (1, 0), dn sites to (0, 1), the rest keep their
    # density, and each spin is then rescaled to its charge:
    # up [1, .75, .75, 0] * 3 / 2.5, dn [0, .25, .25, 1] * 1 / 1.5
    H = _polarized_zgnr2()
    H.set_polarization([0], dn=[3])
    assert np.allclose(H.n, [[1.2, 0.9, 0.9, 0.], [0., 1 / 6, 1 / 6, 2 / 3]])
    assert np.allclose(H.n.sum(1), [3., 1.])

    # Without dn: up [1, .75, .75, .75] * 3 / 3.25, dn [0, .25, .25, .25] * 1 / .75
    H = _polarized_zgnr2()
    H.set_polarization([0])
    assert np.allclose(H.n, [[12 / 13, 9 / 13, 9 / 13, 9 / 13], [0., 1 / 3, 1 / 3, 1 / 3]])
    assert np.allclose(H.n.sum(1), [3., 1.])


def test_polarize_sublattices_polarized():
    # Odd sites are set to (1, 0), even sites to (0, 1), and each spin is then
    # rescaled to its charge: up [0, 1, 0, 1] * 3 / 2, dn [1, 0, 1, 0] * 1 / 2
    H = _polarized_zgnr2()
    H.polarize_sublattices()
    assert np.allclose(H.n, [[0., 1.5, 0., 1.5], [0.5, 0., 0.5, 0.]])
    assert np.allclose(H.n.sum(1), [3., 1.])
