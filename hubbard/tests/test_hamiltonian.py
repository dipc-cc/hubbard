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


# DOS and PDOS average over the k-mesh the object holds

import numpy as np


def _polarized_zgnr():
    # Zigzag ribbon with ferromagnetic edges and one more up than down
    # electron per cell, so that the up and down spectra differ.
    # MonkhorstPack reduces the 16-point mesh by time-reversal symmetry to
    # 9 points of unequal weights (1/16 at k=0 and 1/2, 1/8 elsewhere).
    geom = sisl.geom.zgnr(4)
    H = hh.HubbardHamiltonian(sp2(geom), U=3., q=(4.5, 3.5), nkpt=[16, 1, 1])
    y = geom.xyz[:, 1]
    H.set_polarization([np.argmin(y), np.argmax(y)])
    H.converge(density.calc_n, tol=1e-8, print_info=False)
    return H


def _kmesh_dos(H, egrid, spin, eta, kpoints, weights):
    # Weighted sum of sisl's own DOS over an explicit list of k-points
    dist = sisl.get_distribution('lorentzian', smearing=eta)
    return sum(w * sisl.electron.DOS(egrid, H.H.eigh(k=k, spin=spin), distribution=dist)
               for w, k in zip(weights, kpoints))


def test_dos_pdos_average_over_kmesh():
    H = _polarized_zgnr()
    # Polarized, on a mesh with unequal weights
    assert np.abs(H.n[0] - H.n[1]).max() > 0.1
    assert len(np.unique(np.round(H.mp.weight, 12))) > 1
    # Reference 1: the reduced mesh, read from H.mp
    # Reference 2: the full, unreduced 16-point mesh with equal weights,
    # independent of the reduction and weights in H.mp
    full = sisl.MonkhorstPack(H.H, [16, 1, 1], trs=False)
    assert np.allclose(full.weight, 1 / 16)
    egrid = np.linspace(-4, 4, 161)
    eta = 0.05
    ref = {}
    for s in (0, 1):
        ref[s] = _kmesh_dos(H, egrid, s, eta, H.mp.k, H.mp.weight)
        assert np.allclose(ref[s], _kmesh_dos(H, egrid, s, eta, full.k, full.weight),
                           rtol=1e-10, atol=1e-10)
    # Up and down differ, so a check per spin can tell them apart
    assert np.abs(ref[0] - ref[1]).max() > 1e-2

    for spin, expected in ((0, ref[0]), (1, ref[1]), ([0, 1], ref[0] + ref[1])):
        dos = H.DOS(egrid, eta=eta, spin=spin)
        assert np.allclose(dos, expected, rtol=1e-10, atol=1e-10)
        pdos = H.PDOS(egrid, eta=eta, spin=spin)
        assert pdos.shape == (1, H.sites, len(egrid))
        # sp2 is an orthogonal basis, so the PDOS sums to the DOS
        assert np.allclose(pdos.sum(axis=1)[0], dos, rtol=1e-10, atol=1e-10)
    # The default sums both spins
    assert np.allclose(H.DOS(egrid, eta=eta), ref[0] + ref[1], rtol=1e-10, atol=1e-10)


def test_dos_sum_rule_kmesh():
    H = _polarized_zgnr()
    eta = 0.05
    L = 50.
    egrid = np.linspace(-L, L, 50001)  # step 2e-3 eV, much finer than eta
    eigs = {s: [H.H.eigh(k=k, spin=s) for k in H.mp.k] for s in (0, 1)}
    emax = max(np.abs(e).max() for s in (0, 1) for e in eigs[s])
    # A Lorentzian of width eta centred at e puts a weight
    # (eta/pi) * (1/(L - e) + 1/(L + e)) < (2/pi) * eta / (L - emax) outside
    # [-L, L], so the integral falls short of the orbital count by at most
    # that per state: 6.3e-3 for the 8 orbitals here (|e| < 9.4 eV).
    tail = H.sites * 2 / np.pi * eta / (L - emax)
    for s in (0, 1):
        integral = np.trapezoid(H.DOS(egrid, eta=eta, spin=s), egrid)
        assert H.sites - tail < integral < H.sites
        # The exact integral of each Lorentzian over [-L, L], weighted over k;
        # the trapezoid error at a step of eta/25 is far below 1e-8
        exact = sum(w * np.sum(np.arctan((L - e) / eta) + np.arctan((L + e) / eta)) / np.pi
                    for w, e in zip(H.mp.weight, eigs[s]))
        assert abs(integral - exact) < 1e-8


def test_dos_pdos_molecule_gamma():
    # A molecule (nkpt=[1, 1, 1]) gives the Gamma-point DOS and PDOS, unchanged
    molecule = sisl.geom.agnr(7).tile(2, 0)
    molecule.set_nsc([1, 1, 1])
    H = hh.HubbardHamiltonian(sp2(molecule), U=3.5)
    H.polarize_sublattices()
    H.converge(density.calc_n_insulator, tol=1e-8, print_info=False)
    egrid = np.linspace(-3, 3, 121)
    eta = 0.05
    dist = sisl.get_distribution('lorentzian', smearing=eta)
    dos = 0
    pdos = 0
    for s in (0, 1):
        # Gamma only, as sisl's default k
        dos_s = sisl.electron.DOS(egrid, H.H.eigh(spin=s), distribution=dist)
        ev, evec = H.H.eigh(eigvals_only=False, spin=s)
        pdos_s = sisl.electron.PDOS(egrid, ev, evec.T, distribution=dist)
        assert np.array_equal(H.DOS(egrid, eta=eta, spin=s), dos_s)
        assert np.array_equal(H.PDOS(egrid, eta=eta, spin=s), pdos_s)
        dos = dos + dos_s
        pdos = pdos + pdos_s
    assert np.array_equal(H.DOS(egrid, eta=eta), dos)
    assert np.array_equal(H.PDOS(egrid, eta=eta), pdos)
