import numpy as np
from numpy import einsum, conj
import scipy.linalg as sl
import sisl

import hubbard.density as density
from hubbard import HubbardHamiltonian, sp2


def _mulliken_reference(H, q):
    # Independent reference for calc_n_insulator: solve H c = e S c with scipy,
    # fill the lowest round(q[s]) states of each spin at every k-point and take
    # Mulliken populations Re[c_i^* (S c)_i], weighted by the k-point weights.
    # Also returns the smallest gap above the last filled state, per spin.
    n = np.zeros((H.spin_size, H.sites))
    Etot = 0.
    gap = np.full(H.spin_size, np.inf)
    for k, w in zip(H.mp.k, H.mp.weight):
        Sk = H.H.Sk(k, format='array')
        for s in range(H.spin_size):
            N = int(round(q[s]))
            e, c = sl.eigh(H.H.Hk(k, spin=s, format='array'), Sk)
            occ = c[:, :N]
            n[s] += w * (conj(occ) * (Sk @ occ)).real.sum(1)
            Etot += w * e[:N].sum()
            gap[s] = min(gap[s], e[N] - e[N - 1])
    return n, Etot, gap


def _polarized(H, seed):
    # A polarized density, and the mean-field Hamiltonian it gives
    np.random.seed(seed)
    H.random_density()
    H.update_hamiltonian()
    assert np.abs(H.n[0] - H.n[1]).max() > 0.5


def test_insulator_nonorthogonal_molecule():
    g = sisl.geom.agnr(7).tile(2, 0)
    g.set_nsc([1, 1, 1])
    # Unequal charge per spin, so a mix-up of the two channels shows
    H = HubbardHamiltonian(sp2(g, s1=0.13), U=3.5, q=(15, 13), kT=1e-5)
    assert not H.H.orthogonal
    _polarized(H, 0)

    n, Etot = density.calc_n_insulator(H, H.q)
    n_ref, Etot_ref, gap = _mulliken_reference(H, H.q)
    # The comparison with calc_n below needs a gap in each spin channel
    assert gap.min() > 0.1

    assert n.shape == (2, H.sites)
    # Charge per spin is conserved
    assert np.allclose(n.sum(1), [15, 13], rtol=0, atol=1e-10)
    # Mulliken occupations of the scipy generalized eigenproblem
    assert np.allclose(n, n_ref, rtol=0, atol=1e-10)
    assert np.isclose(Etot, Etot_ref, rtol=0, atol=1e-9)
    # calc_n at kT = 1e-5 eV, far below the gap, fills the same states
    n_fd, Etot_fd = density.calc_n(H, H.q)
    assert np.allclose(n, n_fd, rtol=0, atol=1e-8)
    assert np.isclose(Etot, Etot_fd, rtol=0, atol=1e-7)


def test_insulator_nonorthogonal_periodic_kmesh():
    g = sisl.geom.agnr(7)
    H = HubbardHamiltonian(sp2(g, s1=0.13), U=3.5, nkpt=[5, 1, 1], kT=1e-5)
    assert not H.H.orthogonal
    # Time-reversal reduced mesh with unequal weights
    assert not np.allclose(H.mp.weight, H.mp.weight[0])
    assert np.isclose(H.mp.weight.sum(), 1)
    _polarized(H, 0)

    n, Etot = density.calc_n_insulator(H, H.q)
    n_ref, Etot_ref, gap = _mulliken_reference(H, H.q)
    assert gap.min() > 0.1

    # Charge per spin per unit cell is conserved
    assert np.allclose(n.sum(1), np.round(H.q), rtol=0, atol=1e-10)
    assert np.allclose(n, n_ref, rtol=0, atol=1e-10)
    assert np.isclose(Etot, Etot_ref, rtol=0, atol=1e-9)
    n_fd, Etot_fd = density.calc_n(H, H.q)
    assert np.allclose(n, n_fd, rtol=0, atol=1e-8)
    assert np.isclose(Etot, Etot_fd, rtol=0, atol=1e-7)


def test_insulator_orthogonal_unchanged():
    # For an orthogonal basis the occupations are |c|^2, computed exactly as
    # before the overlap was included, so the output is bit-identical
    g = sisl.geom.agnr(7)
    H = HubbardHamiltonian(sp2(g), U=3.5, nkpt=[5, 1, 1])
    assert H.H.orthogonal
    _polarized(H, 1)

    n, Etot = density.calc_n_insulator(H, H.q)

    n_old = np.zeros((H.spin_size, H.sites))
    Etot_old = 0
    idx = [np.arange(int(round(H.q[s]))) for s in range(H.spin_size)]
    for w, k in zip(H.mp.weight, H.mp.k):
        for s in range(H.spin_size):
            es = H.eigenstate(k, spin=s)
            n_old[s] += einsum('ij,ij->j', conj(es.state[idx[s]]), es.state[idx[s]]).real * w
            Etot_old += es.eig[idx[s]].sum() * w
    assert np.array_equal(n, n_old)
    assert Etot == Etot_old
