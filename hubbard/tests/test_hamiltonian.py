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


# The methods that derive a new HubbardHamiltonian from an existing one.
# Each must build the new reference Hamiltonian (TBHam) from the original's
# TBHam, never from its mean-field H, or update_hamiltonian counts U*n twice.

def _flake(U=3.5, spin='polarized'):
    # Finite zigzag flake: its zigzag edges polarize, so the converged mean-field
    # diagonal differs from the tight-binding one
    g = sisl.geom.zgnr(3).tile(4, 0)
    g.set_nsc([1, 1, 1])
    H = hh.HubbardHamiltonian(sp2(g, spin=spin), U=U)
    if H.spin_size == 2:
        H.set_polarization([0], dn=[g.na - 1])
    H.converge(density.calc_n_insulator, tol=1e-12, print_info=False)
    return H


def _ribbon(U=3., spin='polarized', nk=20):
    # Periodic 3-ZGNR: polarized with antiferromagnetic edges, or unpolarized
    # and doped away from half filling. At half filling an unpolarized
    # bipartite ribbon has n = 1/2 on every site, which would make the
    # mean-field diagonal equal to the reference one.
    if spin == 'unpolarized-doped':
        h = sp2(sisl.geom.zgnr(3), spin='unpolarized')
        q, kT = (h.no / 2 - 0.7,), 0.02
    else:
        h = sp2(sisl.geom.zgnr(3), spin=spin)
        q, kT = (0., 0.), 1e-5
    if U == 'array':
        U = np.full(h.no, 3.)
    H = hh.HubbardHamiltonian(h, U=U, q=q, nkpt=[nk, 1, 1], kT=kT)
    if H.spin_size == 2:
        H.set_polarization([0], dn=[h.no - 1])
    H.converge(density.calc_n, tol=1e-10, print_info=False)
    return H


def _chain(spin='polarized'):
    # Two inequivalent atoms per cell, each with two orbitals (onsite energies
    # eps), so the order of orbitals within an atom matters
    atom = sisl.Atom(6, R=[1.6, 1.6])
    g = sisl.Geometry([[0, 0, 0], [1.42, 0, 0]], atom, lattice=[2.84, 10, 10])
    g.set_nsc([3, 1, 1])
    h = sisl.Hamiltonian(g, spin=spin)
    eps = [[0., 1.5], [0.8, 1.2]]
    t = [[-2.7, -0.4], [-0.4, -1.0]]
    for ia in g:
        o = g.a2o(ia)
        for a in range(2):
            h[o + a, o + a] = eps[ia][a]
        for ja in g.close(ia, R=[0.1, 1.6])[1]:
            p = g.a2o(ja)
            for a in range(2):
                for b in range(2):
                    h[o + a, p + b] = t[a][b]
    return h


def _diag(H, spin):
    return H.tocsr(spin).diagonal()


@pytest.mark.parametrize("system", ['flake', 'unpolarized-doped'])
def test_copy_converged_stays_converged(system):
    if system == 'flake':
        H, calc_n = _flake(), density.calc_n_insulator
    else:
        H, calc_n = _ribbon(spin=system), density.calc_n
    # Precondition: the converged mean-field diagonal differs from the reference
    assert np.abs(_diag(H.H, 0) - _diag(H.TBHam, 0)).max() > 0.1

    Hc = H.copy()
    # Immediately after copy the state is the original's
    for s in range(H.spin_size):
        assert np.array_equal(H.TBHam.tocsr(s).toarray(), Hc.TBHam.tocsr(s).toarray())
        assert np.array_equal(H.H.tocsr(s).toarray(), Hc.H.tocsr(s).toarray())
    assert np.array_equal(H.n, Hc.n)
    assert np.array_equal(H.q, Hc.q)
    assert Hc.Etot == H.Etot

    # Reference: the converged original itself. The first iterate solves the
    # copied H; the second solves the H that update_hamiltonian rebuilt from
    # TBHam, so it catches a double-counted U*n.
    for _ in range(2):
        dn = Hc.iterate(calc_n)
        assert dn < 1e-8
        assert abs(Hc.Etot - H.Etot) < 1e-8
    # The same converge call on both, so the comparison does not depend on how
    # tightly the original was converged
    Hc.converge(calc_n, tol=1e-12, max_iter=300, print_info=False)
    H.converge(calc_n, tol=1e-12, max_iter=300, print_info=False)
    assert abs(Hc.Etot - H.Etot) < 1e-10
    assert np.abs(Hc.n - H.n).max() < 1e-10


@pytest.mark.parametrize("U", ['array', '0-d'])
@pytest.mark.parametrize("spin", ['polarized', 'unpolarized'])
def test_copy_is_independent(spin, U):
    h = sp2(sisl.geom.zgnr(3), spin=spin)
    U = np.full(h.no, 3.) if U == 'array' else np.array(3.)
    H = hh.HubbardHamiltonian(h, U=U, Uij=np.zeros((h.no, h.no)),
                              nkpt=[4, 1, 1], kT=0.01, units='eV-test')
    H.random_density()
    H.iterate(density.calc_n)
    Hc = H.copy()

    assert Hc.spin_size == H.spin_size
    assert Hc.kT == H.kT and Hc.units == H.units
    assert np.array_equal(Hc.mp.k, H.mp.k) and np.array_equal(Hc.mp.weight, H.mp.weight)
    for a in ('n', 'q', 'U', 'Uij'):
        assert np.array_equal(getattr(Hc, a), getattr(H, a))
        assert not np.shares_memory(getattr(Hc, a), getattr(H, a))
    assert Hc.mp is not H.mp

    # Changing or iterating the copy leaves the original untouched
    n, Hd, Td = H.n.copy(), _diag(H.H, 0), _diag(H.TBHam, 0)
    Hc.TBHam[0, 0, 0] = 99.
    Hc.iterate(density.calc_n)
    Hc.H[1, 1, 0] = 99.
    assert np.array_equal(H.n, n)
    assert np.array_equal(_diag(H.H, 0), Hd)
    assert np.array_equal(_diag(H.TBHam, 0), Td)


@pytest.mark.parametrize("method", ['tile', 'repeat'])
@pytest.mark.parametrize("U", [3., 'array'])
@pytest.mark.parametrize("spin", ['polarized', 'unpolarized-doped'])
def test_tile_repeat_converged_stays_converged(method, U, spin):
    nk = 20
    H = _ribbon(U=U, spin=spin, nk=nk)
    # Precondition: the converged mean-field diagonal differs from the reference
    assert np.abs(_diag(H.H, 0) - _diag(H.TBHam, 0)).max() > 0.1

    H2 = getattr(H, method)(2, 0)
    # Both tile and repeat double the lattice vector a1. A Gamma-centred
    # Monkhorst-Pack grid of nk points along a1 folds, under k -> 2k mod 1,
    # onto the Gamma-centred grid of nk/2 points of the doubled cell, each
    # point twice. So nk/2 points on the doubled cell sample the same
    # Brillouin-zone points as nk points on the original one.
    H2.set_kmesh([nk // 2, 1, 1])
    k = sisl.MonkhorstPack(H.geometry, [nk, 1, 1], trs=False).k[:, 0]
    k2 = sisl.MonkhorstPack(H2.geometry, [nk // 2, 1, 1], trs=False).k[:, 0]
    fold = np.sort(np.round(2 * k, 12) % 1)
    assert np.allclose(fold, np.sort(np.repeat(np.round(k2, 12) % 1, 2)))

    # Reference: sisl's tiling of the original reference Hamiltonian
    Tref = getattr(H.TBHam, method)(2, 0)
    for s in range(H.spin_size):
        assert np.array_equal(H2.TBHam.tocsr(s).toarray(), Tref.tocsr(s).toarray())

    # Reference: the converged original, two unit cells of it. The first
    # iterate solves the carried-over H; the second solves the H that
    # update_hamiltonian rebuilt from TBHam, so it catches a double-counted U*n.
    for _ in range(2):
        dn = H2.iterate(density.calc_n)
        assert dn < 1e-8
        assert abs(H2.Etot / 2 - H.Etot) < 1e-8
        assert np.allclose(H2.n.sum(1), 2 * H.q)


@pytest.mark.parametrize("method", ['tile', 'repeat'])
@pytest.mark.parametrize("spin", ['polarized', 'unpolarized'])
def test_tile_repeat_multi_orbital(method, spin):
    # sisl tiles cell by cell and repeats atom by atom, keeping the orbitals
    # of each atom together; n and U must follow the same orbital order
    U_of_eps = {0.: 3., 1.5: 1., 0.8: 2., 1.2: 0.5}
    h = _chain(spin)
    H = hh.HubbardHamiltonian(h, U=np.array([U_of_eps[e] for e in _diag(h, 0)]),
                              nkpt=[12, 1, 1], kT=0.01)
    H.converge(density.calc_n, tol=1e-10, max_iter=500, print_info=False)
    assert np.abs(_diag(H.H, 0) - _diag(H.TBHam, 0)).max() > 0.1

    H2 = getattr(H, method)(2, 0)
    # Reference: U follows the onsite energy of each orbital in sisl's
    # tiled or repeated reference Hamiltonian
    assert np.array_equal(H2.U, [U_of_eps[e] for e in _diag(H2.TBHam, 0)])

    # Reference: the converged original, two unit cells of it (k-mesh as in
    # test_tile_repeat_converged_stays_converged)
    H2.set_kmesh([6, 1, 1])
    for _ in range(2):
        dn = H2.iterate(density.calc_n)
        assert dn < 1e-8
        assert abs(H2.Etot / 2 - H.Etot) < 1e-8


@pytest.mark.parametrize("method", ['tile', 'repeat'])
def test_tile_repeat_Uij_raises(method):
    h = sp2(sisl.geom.zgnr(2))
    H = hh.HubbardHamiltonian(h, U=3., Uij=np.zeros((h.no, h.no)))
    with pytest.raises(NotImplementedError, match="Uij"):
        getattr(H, method)(2, 0)


@pytest.mark.parametrize("method", ['sub', 'remove'])
def test_sub_remove_from_reference(method):
    H = _flake()
    na = H.geometry.na
    removed = [0, 5, 13]
    kept = np.delete(np.arange(na), removed)
    # Precondition: the converged mean-field diagonal differs from the reference
    assert np.abs(_diag(H.H, 0)[kept] - _diag(H.TBHam, 0)[kept]).max() > 0.1

    if method == 'sub':
        Hs = H.sub(kept)
    else:
        Hs = H.remove(removed)

    # Reference: the original TBHam, and sisl's own sub of it
    Tref = H.TBHam.sub(kept)
    for s in range(2):
        assert np.array_equal(_diag(Hs.TBHam, s), _diag(H.TBHam, s)[kept])
        assert np.array_equal(Hs.TBHam.tocsr(s).toarray(), Tref.tocsr(s).toarray())
    assert np.array_equal(Hs.geometry.xyz, H.geometry.xyz[kept])
    # The densities are carried over, renormalized to the charge of the kept atoms
    assert np.allclose(Hs.n.sum(1), Hs.q)

    # The reduced object iterates on its own reference
    Hs.converge(density.calc_n_insulator, tol=1e-10, print_info=False)
    assert np.allclose(Hs.n.sum(1), Hs.q)
    for s in range(2):
        assert np.array_equal(_diag(Hs.TBHam, s), _diag(H.TBHam, s)[kept])


@pytest.mark.parametrize("method", ['sub', 'remove'])
def test_sub_remove_U_Uij(method):
    h = sp2(sisl.geom.zgnr(3))
    removed = [0, 4]
    kept = np.delete(np.arange(h.no), removed)
    U = np.arange(h.no) + 1.
    Uij = np.arange(h.no**2, dtype=float).reshape(h.no, h.no)
    for U_, Uij_, U_ref, Uij_ref in [(3., None, 3., None),
                                     (U, None, U[kept], None),
                                     (3., Uij, 3., Uij[np.ix_(kept, kept)])]:
        H = hh.HubbardHamiltonian(h, U=U_, Uij=Uij_)
        Hs = H.sub(kept) if method == 'sub' else H.remove(removed)
        assert np.array_equal(Hs.U, U_ref)
        if Uij_ref is None:
            assert Hs.Uij is None
        else:
            assert np.array_equal(Hs.Uij, Uij_ref)


@pytest.mark.parametrize("method", ['copy', 'tile', 'repeat', 'sub', 'remove'])
@pytest.mark.parametrize("U", [3., 'array'])
@pytest.mark.parametrize("spin", ['polarized', 'unpolarized'])
def test_derived_scalar_U_unpolarized(method, U, spin):
    h = sp2(sisl.geom.zgnr(2), spin=spin)
    if U == 'array':
        U = np.full(h.no, 3.)
    H = hh.HubbardHamiltonian(h, U=U, nkpt=[4, 1, 1])
    args = {'copy': (), 'tile': (2, 0), 'repeat': (2, 0), 'sub': ([0, 1, 2],), 'remove': ([0],)}
    H2 = getattr(H, method)(*args[method])
    no = {'copy': h.no, 'tile': 2 * h.no, 'repeat': 2 * h.no, 'sub': 3, 'remove': h.no - 1}[method]
    S = 1 if spin == 'unpolarized' else 2
    assert H2.spin_size == S
    assert H2.n.shape == (S, no)
    assert H2.q.shape == (S,)
    assert np.ndim(H2.U) == np.ndim(U)
    H2.iterate(density.calc_n)
    assert np.allclose(H2.n.sum(1), H2.q)
