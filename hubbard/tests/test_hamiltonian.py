import warnings

import pytest

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


def _zgnr_start():
    # Polarized zigzag ribbon started away from its solution, so the SCF
    # loop needs many iterations to reach a tight tolerance
    H = hh.HubbardHamiltonian(sp2(sisl.geom.zgnr(2)), U=3., nkpt=[10, 1, 1])
    H.set_polarization([0], dn=[3])
    return H


def test_converge_warns_at_max_iter():
    # Independent reference: two explicit iterate calls with the mixer
    # converge uses by default, from the same starting density
    ref = _zgnr_start()
    mixer = sisl.mixing.DIISMixer(weight=0.7, history=7)
    for _ in range(2):
        dn_ref = ref.iterate(density.calc_n, mixer=mixer)
    assert dn_ref > 1e-12

    H = _zgnr_start()
    with pytest.warns(UserWarning, match="without converging"):
        dn = H.converge(density.calc_n, tol=1e-12, max_iter=2)
    assert dn == pytest.approx(dn_ref, rel=1e-12, abs=0)
    assert H.Etot == pytest.approx(ref.Etot, rel=1e-12, abs=0)


def test_converge_no_warning_when_converged():
    tol = 1e-6
    # Count the iterations an explicit iterate loop needs to reach tol
    ref = _zgnr_start()
    mixer = sisl.mixing.DIISMixer(weight=0.7, history=7)
    n_iter = 0
    dn_ref = 1.0
    while dn_ref > tol:
        n_iter += 1
        dn_ref = ref.iterate(density.calc_n, mixer=mixer)
        assert n_iter < 1000
    assert n_iter > 2

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Noise from the dependencies, not from converge: sisl's Fermi-Dirac
        # overflows harmlessly far from mu, and older sisl calls deprecated
        # pyparsing functions
        warnings.filterwarnings("ignore", "overflow encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Without an iteration limit
        H = _zgnr_start()
        H.converge(density.calc_n, tol=tol)
        # Converging on the last allowed iteration
        H = _zgnr_start()
        dn = H.converge(density.calc_n, tol=tol, max_iter=n_iter)
    assert dn == pytest.approx(dn_ref, rel=1e-12, abs=0)

    # One iteration fewer stops short of tol and warns
    H = _zgnr_start()
    with pytest.warns(UserWarning, match="without converging"):
        H.converge(density.calc_n, tol=tol, max_iter=n_iter - 1)
