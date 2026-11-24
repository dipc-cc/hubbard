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


def _iterate_q_ribbon(spin='polarized', q=(0., 0.)):
    # Periodic zigzag ribbon (4 sites) from a fixed starting density
    H = hh.HubbardHamiltonian(sp2(sisl.geom.zgnr(2), spin=spin), U=3., q=q, nkpt=[10, 1, 1])
    n = np.array([[0.8, 0.5, 0.5, 0.2], [0.2, 0.5, 0.5, 0.8]])
    H.n = n[:H.spin_size]
    H.update_hamiltonian()
    return H


def _assert_same_iterate(Ha, dna, Hb, dnb):
    np.testing.assert_allclose(Ha.n, Hb.n, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Ha.Etot, Hb.Etot, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dna, dnb, rtol=0, atol=1e-12)


def test_iterate_explicit_q():
    # An explicit q gives the same step as setting H.q and passing q=None,
    # and with an unmixed step the density holds that charge per spin.
    Ha = _iterate_q_ribbon()
    Hb = _iterate_q_ribbon()
    Hb.q = np.array([3., 1.])
    dna = Ha.iterate(density.calc_n, q=[3., 1.], mixer=sisl.mixing.LinearMixer(weight=1.))
    dnb = Hb.iterate(density.calc_n, mixer=sisl.mixing.LinearMixer(weight=1.))
    _assert_same_iterate(Ha, dna, Hb, dnb)
    np.testing.assert_allclose(Ha.n.sum(1), [3., 1.], atol=1e-8)
    np.testing.assert_array_equal(Ha.q, [2., 2.])


def test_iterate_q_none_entry():
    # A None entry takes int(round(H.q[s])); the caller's list is left as it was
    Ha = _iterate_q_ribbon(q=(2.4, 1.6))
    Hb = _iterate_q_ribbon(q=(2.4, 1.6))
    Hb.q = np.array([2., 1.])
    q = [None, 1.]
    dna = Ha.iterate(density.calc_n, q=q, mixer=sisl.mixing.LinearMixer(weight=1.))
    dnb = Hb.iterate(density.calc_n, mixer=sisl.mixing.LinearMixer(weight=1.))
    _assert_same_iterate(Ha, dna, Hb, dnb)
    np.testing.assert_allclose(Ha.n.sum(1), [2., 1.], atol=1e-8)
    assert q == [None, 1.]


def test_iterate_q_unpolarized():
    Ha = _iterate_q_ribbon(spin='unpolarized')
    Hb = _iterate_q_ribbon(spin='unpolarized')
    Hb.q = np.array([3.])
    dna = Ha.iterate(density.calc_n, q=[3.], mixer=sisl.mixing.LinearMixer(weight=1.))
    dnb = Hb.iterate(density.calc_n, mixer=sisl.mixing.LinearMixer(weight=1.))
    _assert_same_iterate(Ha, dna, Hb, dnb)
    np.testing.assert_allclose(Ha.n.sum(1), [3.], atol=1e-8)


@pytest.mark.parametrize("spin, q", [('polarized', [2.]),
                                     ('polarized', [2., 2., 2.]),
                                     ('polarized', 4.),
                                     ('unpolarized', [2., 2.])])
def test_iterate_q_wrong_length(spin, q):
    H = _iterate_q_ribbon(spin=spin)
    with pytest.raises(ValueError, match="one entry per spin channel"):
        H.iterate(density.calc_n, q=q)
