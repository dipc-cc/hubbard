import gc
import warnings

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
    # Uij is not tiled yet: a nonzero Uij raises, an all-zero one is dropped
    h = sp2(sisl.geom.zgnr(2))
    Uij = np.full((h.no, h.no), 0.1)
    H = hh.HubbardHamiltonian(h, U=3., Uij=Uij)
    with pytest.raises(NotImplementedError, match="nonzero Uij"):
        getattr(H, method)(2, 0)
    H = hh.HubbardHamiltonian(h, U=3., Uij=np.zeros((h.no, h.no)))
    assert getattr(H, method)(2, 0).Uij is None


def _two_orbital_molecule(**kwargs):
    # Two atoms with two orbitals each; orbitals 0, 1 on atom 0 and 2, 3 on atom 1
    atom = sisl.Atom(6, R=[1.6, 1.6])
    g = sisl.Geometry([[0, 0, 0], [1.42, 0, 0]], atom, lattice=[10, 10, 10])
    h = sisl.Hamiltonian(g, spin='polarized')
    h[0, 2] = -2.7
    h[1, 3] = -1.
    return hh.HubbardHamiltonian(h, **kwargs)


@pytest.mark.parametrize("method", ['copy', 'tile', 'repeat', 'sub', 'remove'])
def test_derived_Uij_multi_orbital(method):
    # Uij is indexed by orbital, these methods by atom: with more than one
    # orbital per atom only copy carries Uij over
    Uij = np.zeros((4, 4))
    H = _two_orbital_molecule(U=3., Uij=Uij)
    args = {'copy': (), 'tile': (2, 0), 'repeat': (2, 0), 'sub': ([1],), 'remove': ([0],)}[method]
    if method == 'copy':
        assert np.array_equal(H.copy().Uij, Uij)
    else:
        # The message names the method called, also for remove, which uses sub
        with pytest.raises(NotImplementedError, match=rf"\.{method} cannot handle Uij"):
            getattr(H, method)(*args)


@pytest.mark.parametrize("method", ['sub', 'remove'])
@pytest.mark.parametrize("U", [3., 'array'])
def test_sub_remove_multi_orbital(method, U):
    # Reference: the values on the orbitals of the kept atom 1, written out.
    # q is their charge per spin, so renormalization leaves them unchanged.
    n = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.2, 0.1]]
    U = np.array([3., 1., 2., 0.5]) if U == 'array' else U
    H = _two_orbital_molecule(U=U, n=n, q=(1., 0.5))
    Hs = H.sub([1], q=(0.7, 0.3)) if method == 'sub' else H.remove([0], q=(0.7, 0.3))
    assert np.allclose(Hs.n, [[0.3, 0.4], [0.2, 0.1]])
    assert np.array_equal(Hs.U, [2., 0.5] if np.ndim(U) else 3.)
    assert np.array_equal(_diag(Hs.TBHam, 0), _diag(H.TBHam, 0)[2:])


@pytest.mark.parametrize("method", ['copy', 'tile', 'repeat', 'sub', 'remove'])
@pytest.mark.parametrize("empty", [1, 0])
def test_derived_empty_spin_channel(method, empty):
    # Fully polarized: one channel has no charge and no density, and must
    # stay empty rather than become 0/0 = NaN or be given a charge
    h = sp2(sisl.geom.zgnr(2))
    H = hh.HubbardHamiltonian(h, U=3., q=(2., 0.))
    if empty == 0:
        # Spin up empty: the constructor would read q[0] = 0 as "not given"
        H.n = H.n[::-1].copy()
        H.q = H.q[::-1].copy()
    full = 1 - empty
    assert np.array_equal(H.n[empty], np.zeros(h.no))
    kept = [0, 1, 2]
    if method in ('sub', 'remove'):
        q = [0., 0.]
        q[full] = H.n[full, kept].sum()
        H2 = H.sub(kept, q=q) if method == 'sub' else H.remove([3], q=q)
        n_ref = H.n[:, kept]
    else:
        args = {'copy': (), 'tile': (2, 0), 'repeat': (2, 0)}[method]
        H2 = getattr(H, method)(*args)
        n_ref = {'copy': H.n, 'tile': np.tile(H.n, 2), 'repeat': np.repeat(H.n, 2, axis=1)}[method]
    assert np.all(np.isfinite(H2.n))
    assert np.allclose(H2.n, n_ref)
    assert H2.q[empty] == 0.
    assert np.allclose(H2.q, n_ref.sum(1))


def _polarized_molecule():
    # Six-carbon closed molecule, fully polarized: 2 up electrons, no down density
    g = sisl.geom.agnr(5).sub(range(6))
    g.set_nsc([1, 1, 1])
    H = hh.HubbardHamiltonian(sp2(g), U=3., q=(2, 0))
    assert np.array_equal(H.n[1], np.zeros(6))
    return H


@pytest.mark.parametrize("case", ['sub-q', 'sub-default', 'remove-default'])
def test_sub_remove_fill_empty_channel(case):
    # The down channel has no density on the kept atoms 0, 1, 2 but is asked
    # for charge: it starts uniform at q/no instead of 0/0 = NaN
    H = _polarized_molecule()
    if case == 'sub-q':
        Hs, q = H.sub([0, 1, 2], q=(1, 1)), (1., 1.)
    else:
        Hs = H.sub([0, 1, 2]) if case == 'sub-default' else H.remove([3, 4, 5])
        # Default: the three carbon pz electrons of the kept atoms, with the
        # constructor's split q_dn = 3 // 2, q_up = 3 - q_dn
        q = (2., 1.)
    assert np.all(np.isfinite(Hs.n))
    assert np.allclose(Hs.q, q)
    assert np.allclose(Hs.n.sum(1), q)
    assert np.allclose(Hs.n[1], np.full(3, q[1] / 3))


@pytest.mark.parametrize("method", ['sub', 'remove'])
def test_sub_remove_zero_charge_channel_stays_empty(method):
    H = _polarized_molecule()
    Hs = H.sub([0, 1, 2], q=(2, 0)) if method == 'sub' else H.remove([3, 4, 5], q=(2, 0))
    assert np.array_equal(Hs.n[1], np.zeros(3))
    assert np.allclose(Hs.n[0], H.n[0, :3] * 2 / H.n[0, :3].sum())


@pytest.mark.parametrize("method", ['sub', 'remove'])
def test_sub_remove_nan_q_raises(method):
    H = _polarized_molecule()
    with pytest.raises(ValueError, match="non-negative q"):
        if method == 'sub':
            H.sub([0, 1, 2], q=(np.nan, 1))
        else:
            H.remove([3, 4, 5], q=(np.nan, 1))


def test_set_polarization_unchanged():
    # Pins existing behaviour, it does not validate it: the uniform fill of an
    # empty channel belongs to sub/remove only. Maximizing spin up on every
    # site of a q = [4, 4] object leaves no down density to scale to q = 4,
    # which gives NaN, as before this change.
    g = sisl.geom.zgnr(2).tile(2, 0)
    g.set_nsc([1, 1, 1])
    H = hh.HubbardHamiltonian(sp2(g), U=3., q=(4, 4))
    with np.errstate(divide='ignore', invalid='ignore'):
        H.set_polarization(up=range(8))
    np.testing.assert_array_equal(H.n, [[0.5] * 8, [np.nan] * 8])


@pytest.mark.parametrize("method", ['sub', 'remove'])
def test_sub_remove_negative_q_raises(method):
    g = sisl.geom.agnr(5).sub(range(6))
    g.set_nsc([1, 1, 1])
    H = hh.HubbardHamiltonian(sp2(g), U=3.)
    with pytest.raises(ValueError, match="non-negative q"):
        if method == 'sub':
            H.sub([0, 1, 2], q=(-1, 1))
        else:
            H.remove([3, 4, 5], q=(-1, 1))


@pytest.mark.parametrize("method", ['copy', 'tile', 'repeat', 'sub', 'remove'])
@pytest.mark.parametrize("U", ['0-d', 'list'])
def test_derived_owns_U(method, U):
    # A 0-d ndarray or list U is mutable; the derived object must not share it
    h = sp2(sisl.geom.zgnr(2))
    U = np.array(3.) if U == '0-d' else [3.] * h.no
    H = hh.HubbardHamiltonian(h, U=U)
    args = {'copy': (), 'tile': (2, 0), 'repeat': (2, 0), 'sub': ([0, 1, 2],), 'remove': ([0],)}[method]
    H2 = getattr(H, method)(*args)
    assert H2.U is not H.U
    if np.ndim(H2.U):
        H2.U[0] = 7.
    else:
        H2.U[...] = 7.
    assert np.all(np.asarray(H.U) == 3.)


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


# DOS and PDOS average over the k-mesh the object holds


def test_dos_pdos_chain_analytic():
    # One-orbital chain with hopping -t: its Lorentzian-broadened DOS per spin
    # is -Im G(z) / pi with G(z) = 1 / sqrt(z^2 - 4 t^2), z = E + i eta.
    # Written as 1 / (sqrt(z - 2t) * sqrt(z + 2t)) with principal roots, both
    # factors lie in the upper half plane for eta > 0, which picks the
    # retarded branch, Im G < 0, inside and outside the band.
    t = 2.7
    eta = 0.1
    geom = sisl.Geometry([0, 0, 0], sisl.Atom(6, R=1.5),
                         lattice=sisl.Lattice([1.42, 10, 10], nsc=[3, 1, 1]))
    tb = sisl.Hamiltonian(geom, spin=sisl.Spin('polarized'))
    tb.construct([[0.1, 1.5], [0., -t]])
    H = hh.HubbardHamiltonian(tb, U=0., nkpt=[2000, 1, 1])
    egrid = np.linspace(-7, 7, 141)
    z = egrid + 1j * eta
    ref = -np.imag(1 / (np.sqrt(z - 2 * t) * np.sqrt(z + 2 * t))) / np.pi
    for s in (0, 1):
        assert np.allclose(H.DOS(egrid, eta=eta, spin=s), ref, rtol=0, atol=1e-10)
        assert np.allclose(H.PDOS(egrid, eta=eta, spin=s)[0, 0], ref, rtol=0, atol=1e-10)
    assert np.allclose(H.DOS(egrid, eta=eta), 2 * ref, rtol=0, atol=1e-10)


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


# PDOS projects with the overlap (Mulliken) in a non-orthogonal basis


def _polarized_zgnr_nonorthogonal():
    # As _polarized_zgnr, with a first-neighbour overlap s1
    geom = sisl.geom.zgnr(4)
    H = hh.HubbardHamiltonian(sp2(geom, s1=0.13), U=3., q=(4.5, 3.5), nkpt=[16, 1, 1])
    y = geom.xyz[:, 1]
    H.set_polarization([np.argmin(y), np.argmax(y)])
    H.converge(density.calc_n, tol=1e-8, print_info=False)
    return H


def test_pdos_mulliken_nonorthogonal():
    import scipy.linalg as sla
    H = _polarized_zgnr_nonorthogonal()
    assert not H.H.orthogonal
    assert np.abs(H.n[0] - H.n[1]).max() > 0.1
    assert len(np.unique(np.round(H.mp.weight, 12))) > 1
    egrid = np.linspace(-4, 4, 161)
    eta = 0.05

    # Reference on the full, unreduced 16-point mesh with equal weights: the
    # generalized eigenproblem H(k) c = e S(k) c solved by scipy, the Mulliken
    # weight Re[conj(c_i) (S(k) c)_i] of each state on each orbital, and a
    # Lorentzian (eta / pi) / ((E - e)^2 + eta^2) written out here.
    full = sisl.MonkhorstPack(H.H, [16, 1, 1], trs=False)
    lorentz = lambda e: (eta / np.pi) / ((egrid[None, :] - e[:, None]) ** 2 + eta ** 2)
    mulliken = {}
    naive = {}
    for s in (0, 1):
        mulliken[s] = 0
        naive[s] = 0
        for w, k in zip(full.weight, full.k):
            Hk = H.H.Hk(k=k, spin=s, format='array')
            Sk = H.H.Sk(k=k, format='array')
            e, c = sla.eigh(Hk, Sk)
            weight = (np.conj(c) * (Sk @ c)).real  # (orbital, state)
            mulliken[s] = mulliken[s] + w * weight @ lorentz(e)
            naive[s] = naive[s] + w * (np.abs(c) ** 2) @ lorentz(e)
    # The overlap matters: the Mulliken and |c|^2 projections differ
    assert np.abs(mulliken[0] - naive[0]).max() > 1e-2
    assert np.abs(mulliken[0] - mulliken[1]).max() > 1e-2

    for spin, expected in ((0, mulliken[0]), (1, mulliken[1]),
                           ([0, 1], mulliken[0] + mulliken[1])):
        pdos = H.PDOS(egrid, eta=eta, spin=spin)
        assert pdos.shape == (1, H.sites, len(egrid))
        assert np.allclose(pdos[0], expected, rtol=0, atol=1e-10)
        # Summed over orbitals, the PDOS is the DOS
        dos = H.DOS(egrid, eta=eta, spin=spin)
        assert np.allclose(pdos[0].sum(axis=0), dos, rtol=0, atol=1e-10)


def test_pdos_orthogonal_unchanged():
    # An orthogonal basis projects with |c|^2, through sisl's PDOS without an
    # overlap, exactly as before the overlap was included
    H = _polarized_zgnr()
    assert H.H.orthogonal
    egrid = np.linspace(-4, 4, 161)
    eta = 0.05
    dist = sisl.get_distribution('lorentzian', smearing=eta)
    for spin in (0, 1, [0, 1]):
        ref = 0
        for s in ([spin] if isinstance(spin, int) else spin):
            for w, k in zip(H.mp.weight, H.mp.k):
                ev, evec = H.H.eigh(k=k, eigvals_only=False, spin=s)
                ref += w * sisl.physics.electron.PDOS(egrid, ev, evec.T, distribution=dist)
        assert np.array_equal(H.PDOS(egrid, eta=eta, spin=spin), ref)


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


@pytest.mark.parametrize("spin, Hq, q, expected", [('polarized', (2.4, 1.6), [None, None], [2.4, 1.6]),
                                                   ('polarized', (2.4, 1.6), [None, 1.], [2.4, 1.]),
                                                   ('unpolarized', (1.5,), [None], [1.5])])
def test_iterate_q_none_entry(spin, Hq, q, expected):
    # A None entry takes H.q[s] unrounded (fractional and half-integer counts
    # are kept); the caller's list is left as it was
    Ha = _iterate_q_ribbon(spin=spin, q=Hq)
    Hb = _iterate_q_ribbon(spin=spin, q=Hq)
    Hb.q = np.array(expected)
    q_in = list(q)
    dna = Ha.iterate(density.calc_n, q=q, mixer=sisl.mixing.LinearMixer(weight=1.))
    dnb = Hb.iterate(density.calc_n, mixer=sisl.mixing.LinearMixer(weight=1.))
    _assert_same_iterate(Ha, dna, Hb, dnb)
    np.testing.assert_allclose(Ha.n.sum(1), expected, atol=1e-8)
    np.testing.assert_array_equal(Ha.q, Hq)
    assert q == q_in


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


@pytest.mark.parametrize("q", [[float('nan'), 2.], [-1., 5.]])
def test_iterate_q_negative_or_nan_raises(q):
    # Checked before the density solver runs: sisl's fermi_level loops forever on NaN
    def calc_n_not_reached(H, q):
        raise AssertionError("calc_n_method was called")

    H = _iterate_q_ribbon()
    with pytest.raises(ValueError, match="non-negative q"):
        H.iterate(calc_n_not_reached, q=q)


def _zgnr_start():
    # Polarized zigzag ribbon started away from its solution, so the SCF
    # loop needs many iterations to reach a tight tolerance
    H = hh.HubbardHamiltonian(sp2(sisl.geom.zgnr(2)), U=3., nkpt=[10, 1, 1])
    H.set_polarization([0], dn=[3])
    return H


def test_converge_warns_at_max_iter():
    # Consistency check against iterate, which converge calls: two explicit
    # iterate calls with converge's default mixer, from the same start, must
    # give the dn converge returns
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


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("max_iter", [None, 5])
def test_converge_raises_on_non_finite_dn(value, max_iter):
    # A NaN dn would end the loop as if converged (NaN > tol is False), and
    # an infinite one would keep it iterating; both must stop at once
    calls = []

    def bad_density(H, q):
        calls.append(1)
        return np.full_like(H.n, value), 0.

    H = _zgnr_start()
    with pytest.raises(FloatingPointError, match="not finite"):
        H.converge(bad_density, max_iter=max_iter)
    assert len(calls) == 1


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


def _initspin_hamiltonian():
    # zgnr(2): four carbon sites, one pz orbital each, with distinct polarizations
    H = hh.HubbardHamiltonian(sp2(sisl.geom.zgnr(2)), U=3.)
    H.n = np.array([[0.61, 0.32, 0.47, 0.55], [0.39, 0.68, 0.51, 0.41]])
    return H


def _read_initspin(fn):
    # parse the Spin.Total value and the (index, polarization) pairs of the DM.InitSpin block
    total, block, inside = None, [], False
    with open(fn) as f:
        for line in f:
            words = line.split()
            if line.startswith('Spin.Total'):
                total = float(words[1])
            elif line.startswith('%block DM.InitSpin'):
                inside = True
            elif line.startswith('%endblock DM.InitSpin'):
                inside = False
            elif inside:
                block.append((int(words[0]), float(words[1])))
    return total, block


def test_write_initspin_ext_geom_permuted(tmp_path):
    # ext_geom holds internal atoms 2, 3 and 0 in that order (atom 1 is missing),
    # plus an atom that is not in the Hamiltonian
    H = _initspin_hamiltonian()
    extra = sisl.Geometry([[30., 0., 0.]], atoms=sisl.Atom(6))
    ext = H.geometry.sub([2, 3, 0]).add(extra)
    fn = tmp_path / 'init.fdf'
    with pytest.warns(UserWarning, match='no match in ext_geom'):
        H.write_initspin(fn, ext_geom=ext)
    total, block = _read_initspin(fn)

    # reference: match each external atom to an internal one by its coordinates
    pol = H.n[0] - H.n[1]
    expected = {}
    for ia, xyz in enumerate(ext.xyz):
        d = np.linalg.norm(H.geometry.xyz - xyz, axis=1)
        if d.min() < 1e-6:
            expected[ia + 1] = pol[np.argmin(d)]
    assert sorted(i for i, _ in block) == sorted(expected) == [1, 2, 3]
    for i, p in block:
        assert p == pytest.approx(expected[i], abs=1e-6)
    # Spin.Total sums the polarizations of the internal atoms present in ext_geom only
    assert total == pytest.approx(pol[[2, 3, 0]].sum(), abs=1e-6)


def test_write_initspin_default_unchanged(tmp_path):
    # Stored reference: the text written by the code before the ext_geom fix (pins behavior)
    expected = (
        'Spin.Fix True\n'
        'Spin.Total -0.040000\n'
        '%block DM.InitSpin\n'
        '     1   0.220000 #  C    0.0000   10.0000   10.0000\n'
        '     2  -0.360000 #  C    0.0000   12.8400   10.0000\n'
        '     3  -0.040000 #  C    1.2298   10.7100   10.0000\n'
        '     4   0.140000 #  C    1.2298   12.1300   10.0000\n'
        '%endblock DM.InitSpin\n\n'
    )
    H = _initspin_hamiltonian()
    fn = tmp_path / 'init.fdf'
    H.write_initspin(fn)
    assert fn.read_text() == expected


def test_write_initspin_closes_file(tmp_path):
    H = _initspin_hamiltonian()
    fn = tmp_path / 'init.fdf'
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        H.write_initspin(fn)
        H.write_initspin(fn, ext_geom=H.geometry)
        gc.collect()
    assert not [w for w in record if issubclass(w.category, ResourceWarning)]


def test_write_initspin_ext_geom_no_match(tmp_path):
    # no atom in common: an empty block with Spin.Total 0 would pin SIESTA's moment to zero
    H = _initspin_hamiltonian()
    fn = tmp_path / 'init.fdf'
    with pytest.raises(ValueError, match='no atom of ext_geom'):
        H.write_initspin(fn, ext_geom=H.geometry.translate([50., 0., 0.]))
    assert not fn.exists()


def test_write_initspin_ext_geom_unmatched_warns(tmp_path):
    # atom 0 of ext_geom sits at a periodic image of internal atom 0, which overlap does not search
    H = _initspin_hamiltonian()
    ext = H.geometry.copy()
    ext.xyz[0] += ext.cell[0]
    fn = tmp_path / 'init.fdf'
    with pytest.warns(UserWarning, match='1 atoms of the Hamiltonian geometry have no match'):
        H.write_initspin(fn, ext_geom=ext)
    total, block = _read_initspin(fn)
    pol = H.n[0] - H.n[1]
    assert [i for i, _ in block] == [2, 3, 4]
    assert total == pytest.approx(pol[1:].sum(), abs=1e-6)


@pytest.mark.parametrize('case', ['two_external_near_one', 'eps_spans_neighbours'])
def test_write_initspin_ext_geom_ambiguous(tmp_path, case):
    H = _initspin_hamiltonian()
    if case == 'two_external_near_one':
        # a second external atom 0.05 Ang from internal atom 0
        near = sisl.Geometry([H.geometry.xyz[0] + [0.05, 0., 0.]], atoms=sisl.Atom(6))
        ext, eps = H.geometry.add(near), 0.1
    else:
        # eps larger than the 1.42 Ang bond: every atom matches its neighbours too
        ext, eps = H.geometry, 1.5
    fn = tmp_path / 'init.fdf'
    with pytest.raises(ValueError, match='more than one atom'):
        H.write_initspin(fn, ext_geom=ext, eps=eps)
    assert not fn.exists()
