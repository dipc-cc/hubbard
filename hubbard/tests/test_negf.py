import math

import numpy as np
import pytest
import sisl

import hubbard.negf as negf_module
from hubbard import HubbardHamiltonian, NEGF, sp2


def _square_strip(spin):
    """ Square-lattice electrode cell (2 sites, periodic along a2) and a device of 3 cells """
    geom = sisl.Geometry([[0., 0., 0.], [0., 1., 0.]], sisl.Atom(1, R=1.1), lattice=[1., 2., 10.])
    geom.set_nsc([3, 3, 1])
    elec = sisl.Hamiltonian(geom, spin=spin)
    onsite = [0.] * elec.spin.spinor
    hop = [-1.] * elec.spin.spinor
    elec.construct([(0.1, 1.1), (onsite, hop)])
    dev = elec.tile(3, 0)
    dev.set_nsc([1, 3, 1])
    # Break the sublattice symmetry so that Ef = 0 is not special
    for io, e in zip(range(dev.no), [0.3, -0.2, 0.1, 0.4, -0.3, 0.2]):
        for s in range(dev.spin.spinor):
            dev[io, io, s] = e
    return elec, dev


def _open_system(n, spin):
    """ Device with a time-reversal reduced, unequal-weight k-mesh and RecursiveSI electrodes """
    elec, dev = _square_strip(spin)
    H = HubbardHamiltonian(dev, n=n, U=2., nkpt=[1, 5, 1], kT=0.025)
    H.update_hamiltonian()
    SE = [sisl.RecursiveSI(elec, '-A'), sisl.RecursiveSI(elec, '+A')]
    elec_idx = [range(elec.no), range(dev.no - elec.no, dev.no)]
    return H, SE, NEGF(H, SE, elec_idx), elec_idx


@pytest.fixture
def fermi_path(monkeypatch):
    """ Record the device Fermi level of every density pass of `NEGF.calc_n_open`

    Every pass evaluates the Green function on the equilibrium contour shifted
    by the current Ef, so the shift of its first point is that pass's Ef.
    """
    _G = negf_module._G
    calls = []

    def recorder(e, *args):
        calls.append(e)
        return _G(e, *args)
    monkeypatch.setattr(negf_module, '_G', recorder)

    def path(negf, H):
        c0 = negf.CC_eq[0, 0]
        Ef = [e.real - c0.real for e in calls if e.imag == c0.imag]
        calls.clear()
        # one call per spin and k-point in each pass
        return np.array(Ef[::H.spin_size * len(H.mp.k)])
    return path


def _polarized_density():
    rng = np.random.default_rng(1)
    n = rng.uniform(0.2, 0.8, size=(2, 6))
    n[0, :3] += 0.3
    return n


def test_fermi_step_weighted_dos(fermi_path):
    # The first Lorentzian step of the Fermi-level search, against the
    # density of states at the starting Ef computed here from sisl's own
    # self-energies and a matrix inverse, weighted over spins and k-points.
    H, SE, negf, elec_idx = _open_system(_polarized_density(), 'polarized')
    assert H.spin_size == 2
    assert not np.allclose(H.mp.weight, H.mp.weight[0])
    q = H.q + np.array([0.2, 0.1])
    Ef0 = negf.Ef
    assert Ef0 == 0.
    eta = negf.eta

    # Charge mismatch at Ef0, from the solver's own first density pass
    # (ntot starts at -1, so only that pass exceeds this qtol)
    ni0, _ = negf.calc_n_open(H, q, qtol=q.sum() + 0.5)
    assert negf.Ef == Ef0
    dq = ni0.sum() - q.sum()
    assert abs(dq) > 0.1
    assert len(fermi_path(negf, H)) == 1

    # -Im Tr G(Ef0 + i eta) / pi, summed over spins and k with the mesh weights
    E = Ef0 + 1j * eta
    dos = 0.
    for spin in range(2):
        for w, k in zip(H.mp.weight, H.mp.k):
            M = E * np.eye(H.sites) - H.H.Hk(k=k, spin=spin, format='array')
            for se, idx in zip(SE, elec_idx):
                idx = np.asarray(idx)
                M[np.ix_(idx, idx)] -= se.self_energy(E, k=k, spin=spin)
            dos -= w * np.trace(np.linalg.inv(M)).imag / math.pi
    x = dq / dos
    if abs(x) < 0.45:
        Ef1 = Ef0 - eta * math.tan(x * math.pi) * 0.5
    else:
        Ef1 = Ef0 - eta * math.tan(math.pi / 2 - math.atan(1 / (x * math.pi))) * 0.5

    negf.Ef = Ef0
    negf.calc_n_open(H, q, qtol=1e-6)
    path = fermi_path(negf, H)
    assert path[0] == pytest.approx(Ef0, abs=1e-12)
    assert path[1] == pytest.approx(Ef1, abs=1e-10)


def test_fermi_path_spin_swap(fermi_path):
    # Swapping up and down in the device leaves the Fermi-level search unchanged
    n = _polarized_density()
    H, _, negf, _ = _open_system(n, 'polarized')
    Hs, _, negfs, _ = _open_system(n[::-1].copy(), 'polarized')
    q = H.q + np.array([0.2, 0.1])
    qtol = 1e-7

    ni, Etot = negf.calc_n_open(H, q, qtol=qtol)
    path = fermi_path(negf, H)
    nis, Etots = negfs.calc_n_open(Hs, q[::-1], qtol=qtol)
    paths = fermi_path(negfs, Hs)

    assert len(path) > 2
    assert len(path) == len(paths)
    assert np.allclose(path, paths, rtol=0, atol=1e-10)
    assert np.allclose(ni, nis[::-1], rtol=0, atol=1e-8)
    assert Etot == pytest.approx(Etots, abs=1e-8)


def test_fermi_path_unpolarized_polarized(fermi_path):
    # An unpolarized device and the same device polarized at zero polarization
    # take the same Fermi path: dq and the DOS both count one spin channel
    # (unpolarized) or both spin channels (polarized), so their ratio agrees
    n0 = _polarized_density()[:1]
    Hu, _, negfu, _ = _open_system(n0, 'unpolarized')
    Hp, _, negfp, _ = _open_system(np.vstack([n0, n0]), 'polarized')
    assert Hu.spin_size == 1 and Hp.spin_size == 2
    qtol = 1e-7
    qu = Hu.q + 0.15
    qp = Hp.q + 0.15
    assert qp.sum() == pytest.approx(2 * qu.sum())

    nu, Eu = negfu.calc_n_open(Hu, qu, qtol=qtol)
    pathu = fermi_path(negfu, Hu)
    # the polarized charge criterion is on both spins, so give it twice qtol
    npol, Ep = negfp.calc_n_open(Hp, qp, qtol=2 * qtol)
    pathp = fermi_path(negfp, Hp)

    assert len(pathu) > 2
    assert len(pathu) == len(pathp)
    assert np.allclose(pathu, pathp, rtol=0, atol=1e-10)
    assert negfu.Ef == pytest.approx(negfp.Ef, abs=1e-10)
    assert np.allclose(npol, np.vstack([nu, nu]), rtol=0, atol=qtol)
    assert Eu == pytest.approx(Ep, abs=1e-6)


@pytest.mark.parametrize("qtol", [1e-5, 1e-8])
def test_fermi_search_converges_across_gap(monkeypatch, qtol):
    # A gapped 7-AGNR segment between wide-band electrodes, from starting Ef
    # on both sides of and inside the gap (midgap ~0.43 eV). Interpolating
    # through all earlier points stalled for some starts (thousands of density
    # passes); the bracketed search needs at most 22 passes here, and the
    # budget leaves about a factor two of margin.
    budget = 40

    class Budget(Exception):
        pass

    geom = sisl.geom.agnr(7).tile(2, 0)
    geom.set_nsc([1, 1, 1])
    H = HubbardHamiltonian(sp2(geom), U=3., kT=0.025)
    H.set_polarization([0], dn=[-1])
    nl = 7
    elec_idx = [range(nl), range(H.sites - nl, H.sites)]
    negf = NEGF(H, [sisl.WideBandSE(nl, 0.), sisl.WideBandSE(nl, 0.)], elec_idx)
    negf.eta = 1e-2
    c0 = negf.CC_eq[0, 0]

    _G = negf_module._G
    passes = [0]

    def counter(e, *args):
        if e.imag == c0.imag:
            passes[0] += 1
            if passes[0] > budget * H.spin_size:
                raise Budget
        return _G(e, *args)
    monkeypatch.setattr(negf_module, '_G', counter)

    stalled = []
    for Ef0 in [-1., -0.5, 0., 0.2, 0.5, 0.7, 0.8, 1., 1.2, 1.5]:
        negf.Ef = Ef0
        passes[0] = 0
        try:
            ni, _ = negf.calc_n_open(H, H.q, qtol=qtol)
        except Budget:
            stalled.append(Ef0)
            continue
        assert abs(ni.sum() - H.q.sum()) <= qtol
    assert stalled == []
