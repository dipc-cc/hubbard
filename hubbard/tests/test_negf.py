import math

import numpy as np
import pytest
import sisl

import hubbard.density as density
import hubbard.negf as negf_module
from hubbard import HubbardHamiltonian, sp2, NEGF

# The default equilibrium contour (hubbard/EQCONTOUR) is a 300 K contour:
# its Matsubara poles are spaced 2*pi*kT = 0.1624267490396 eV apart.
# The molecule below is used at that kT for consistency; its gap is wide
# enough that the closed-molecule comparison does not depend on it.
kT_contour = 0.1624267490396 / (2 * np.pi)


def _molecule(kT):
    # A finite 7-AGNR segment, 28 sites, with a HOMO-LUMO gap of 0.76 eV
    g = sisl.geom.agnr(7).tile(2, 0)
    g.set_nsc([1, 1, 1])
    return HubbardHamiltonian(sp2(g), U=0., kT=kT)


def test_wideband_zero_eta_is_closed_molecule():
    # Electrodes with eta = 0 leave the molecule closed: the NEGF density
    # and energy must be those of diagonalization. Across the 0.76 eV gap
    # thermal occupation is negligible, so this does not probe kT
    H = _molecule(kT_contour)
    n_ref, Etot_ref = density.calc_n(H, H.q)

    negf = NEGF(H, [sisl.WideBandSE(2, 0.), sisl.WideBandSE(2, 0.)], [[0, 1], [26, 27]])
    negf.Ef = H.find_midgap()
    n, Etot = negf.calc_n_open(H, H.q)

    assert n.shape == n_ref.shape
    assert np.allclose(n, n_ref, rtol=0, atol=1e-6)
    assert Etot == pytest.approx(Etot_ref, abs=1e-4)


class _WideBandNoOverride(sisl.WideBandSE):
    pass


class _WideBandNoOverrideChild(_WideBandNoOverride):
    pass


class _WideBandOverride(sisl.WideBandSE):
    # A user's wide-band electrode with its own, energy-independent coupling
    def self_energy(self, *args, **kwargs):
        return np.diag(np.repeat(-0.4j, len(self)))


def _instance_override(no, eta):
    # A user's wide-band electrode whose method is replaced on the instance
    se = sisl.WideBandSE(no, eta)
    se.self_energy = lambda *args, **kwargs: np.diag(np.repeat(-0.9j, no))
    return se


@pytest.mark.parametrize("make, coupling", [
    (sisl.WideBandSE, 0.1),
    (_WideBandNoOverride, 0.1),
    (_WideBandNoOverrideChild, 0.1),
    (_WideBandOverride, 0.4),
    (_instance_override, 0.9),
], ids=["WideBandSE", "subclass", "subsubclass", "subclass-override", "instance-override"])
def test_wideband_coupling(make, coupling):
    # A WideBandSE(n, eta) couples its orbitals with -i*eta*I at every energy,
    # whatever the installed sisl version; an override of self_energy, in a
    # subclass or on the instance, couples with what it returns
    eta = 0.1
    H = _molecule(kT_contour)
    no = len(H.H)
    negf = NEGF(H, [make(no, eta)], [np.arange(no)])

    se_ref = -1j * coupling * np.eye(no)
    for spin in range(H.spin_size):
        assert np.array_equal(negf._ef_SE[spin][0][0], se_ref)
        for se_cc in negf._cc_eq_SE[spin][0][0]:
            assert np.array_equal(se_cc[0], se_ref)

    # With every orbital coupled, the device Green function is
    # (E + i*(eta_dos + coupling) - H)^-1, so each eigenstate contributes a
    # Lorentzian of half-width eta_dos + coupling to the DOS
    E = np.linspace(-1, 1, 21)
    eta_dos = 0.01
    gamma = eta_dos + coupling
    eig, vec = np.linalg.eigh(H.H.Hk(spin=0, format='array'))
    lorentz = gamma / np.pi / ((E[:, None] - eig[None, :]) ** 2 + gamma ** 2)
    pdos_ref = lorentz @ (np.abs(vec.T) ** 2)

    assert np.allclose(negf.DOS(H, E, spin=0, eta=eta_dos), lorentz.sum(1), rtol=1e-10, atol=0)
    assert np.allclose(negf.PDOS(H, E, spin=0, eta=eta_dos), pdos_ref, rtol=1e-10, atol=1e-14)


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


def _molecule_with_leads(spin):
    # The 28-site 7-AGNR segment of _molecule, at U = 0, with a wide-band
    # electrode of coupling eta = 0.1 eV on each pair of end sites
    g = sisl.geom.agnr(7).tile(2, 0)
    g.set_nsc([1, 1, 1])
    H = HubbardHamiltonian(sp2(g, spin=spin), U=0., kT=kT_contour)
    elec_idx = [[0, 1], [26, 27]]
    negf = NEGF(H, [sisl.WideBandSE(2, 0.1), sisl.WideBandSE(2, 0.1)], elec_idx)
    return H, negf, elec_idx


def test_dos_pdos_unpolarized():
    # The DOS of the open molecule is -Im Tr[G]/pi with
    # G = (E + i*eta - H_eff)^-1 and H_eff = H - i*0.1 on the electrode
    # sites. With lambda the eigenvalues of the non-Hermitian H_eff,
    # Tr[G] = sum 1/(E + i*eta - lambda); the PDOS is -Im diag(G)/pi
    H, negf, elec_idx = _molecule_with_leads('unpolarized')
    assert H.spin_size == 1
    E = np.linspace(-3, 3, 31)
    eta = 0.01

    Heff = H.H.Hk(format='array').astype(np.complex128)
    for idx in elec_idx:
        Heff[idx, idx] -= 0.1j
    lam = np.linalg.eigvals(Heff)
    dos_ref = -(1 / (E[:, None] + 1j * eta - lam[None, :])).sum(1).imag / np.pi
    pdos_ref = np.array([-np.linalg.inv((e + 1j * eta) * np.eye(len(Heff)) - Heff).diagonal().imag
                         for e in E]) / np.pi

    dos0 = negf.DOS(H, E, spin=0, eta=eta)
    assert np.allclose(dos0, dos_ref, rtol=1e-10, atol=1e-12)
    # Both spin indices are the single channel; the default sums the two
    assert np.array_equal(negf.DOS(H, E, spin=1, eta=eta), dos0)
    assert np.allclose(negf.DOS(H, E, eta=eta), 2 * dos0, rtol=1e-14, atol=0)

    pdos0 = negf.PDOS(H, E, spin=0, eta=eta)
    assert pdos0.shape == (len(E), len(H.H))
    assert np.allclose(pdos0, pdos_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(pdos0.sum(1), dos0, rtol=1e-10, atol=1e-12)
    assert np.array_equal(negf.PDOS(H, E, spin=1, eta=eta), pdos0)
    assert np.allclose(negf.PDOS(H, E, eta=eta), 2 * pdos0, rtol=1e-14, atol=0)


def test_dos_pdos_unpolarized_equals_polarized_at_zero_polarization():
    # At U = 0 the polarized molecule has the unpolarized H in both spin
    # channels, so each channel's DOS and PDOS are the unpolarized ones
    Hu, negf_u, _ = _molecule_with_leads('unpolarized')
    Hp, negf_p, _ = _molecule_with_leads('polarized')
    assert Hp.spin_size == 2
    Hk = Hu.H.Hk(format='array')
    for s in range(2):
        assert np.array_equal(Hp.H.Hk(spin=s, format='array'), Hk)
    E = np.linspace(-3, 3, 31)

    dos_u = negf_u.DOS(Hu, E, spin=0)
    pdos_u = negf_u.PDOS(Hu, E, spin=0)
    for s in range(2):
        assert np.allclose(negf_p.DOS(Hp, E, spin=s), dos_u, rtol=1e-12, atol=0)
        assert np.allclose(negf_p.PDOS(Hp, E, spin=s), pdos_u, rtol=1e-12, atol=1e-15)
    assert np.allclose(negf_p.DOS(Hp, E), negf_u.DOS(Hu, E), rtol=1e-12, atol=0)
    assert np.allclose(negf_p.PDOS(Hp, E), negf_u.PDOS(Hu, E), rtol=1e-12, atol=1e-15)


def test_dos_pdos_unpolarized_recursive_electrodes():
    # An unpolarized ZGNR(2) device, three electrode cells long, between two
    # semi-infinite ZGNR(2) electrodes. NEGF shifts each electrode by its own
    # Fermi level (non-zero with the t2, t3 hoppings); the reference applies
    # that shift to an independent copy of the electrode, takes sisl's
    # RecursiveSI self-energies and inverts the device matrix directly
    kT = 0.025
    He = sp2(sisl.geom.zgnr(2), spin='unpolarized')
    elec = HubbardHamiltonian(He, U=0., nkpt=[30, 1, 1], kT=kT)
    Ef_elec = elec.fermi_level(q=elec.q, distribution=sisl.get_distribution('fermi_dirac', smearing=kT))
    assert abs(Ef_elec) > 0.1
    He_ref = He.copy()
    He_ref.shift(-Ef_elec)
    SE_ref = [sisl.RecursiveSI(He_ref, '-A'), sisl.RecursiveSI(He_ref, '+A')]

    HC = He.tile(3, axis=0)
    HC.set_nsc([1, 1, 1])
    elec_idx = [np.arange(len(He)), np.arange(len(HC) - len(He), len(HC))]
    H = HubbardHamiltonian(HC, U=0., kT=kT)
    assert H.spin_size == 1
    negf = NEGF(H, [(elec, '-A'), (elec, '+A')], elec_idx)

    E = np.linspace(-1, 1, 11)
    eta = 0.01
    HCa = HC.Hk(format='array')
    pdos_ref = []
    for e in E:
        A = (e + 1j * eta) * np.eye(len(HCa)) - HCa
        for idx, se in zip(elec_idx, SE_ref):
            A[np.ix_(idx, idx)] -= se.self_energy(e)
        pdos_ref.append(-np.linalg.inv(A).diagonal().imag / np.pi)
    pdos_ref = np.array(pdos_ref)

    dos0 = negf.DOS(H, E, spin=0, eta=eta)
    assert np.allclose(dos0, pdos_ref.sum(1), rtol=1e-8, atol=1e-10)
    assert np.allclose(negf.DOS(H, E, eta=eta), 2 * dos0, rtol=1e-14, atol=0)
    pdos0 = negf.PDOS(H, E, spin=0, eta=eta)
    assert np.allclose(pdos0, pdos_ref, rtol=1e-8, atol=1e-10)
    assert np.allclose(negf.PDOS(H, E, eta=eta), 2 * pdos0, rtol=1e-14, atol=0)


@pytest.mark.parametrize("negf_spin, H_spin", [
    ('polarized', 'unpolarized'),
    ('unpolarized', 'polarized'),
])
def test_dos_pdos_spin_mismatch(negf_spin, H_spin):
    # The electrode self-energies are set up per spin channel of the device
    # NEGF was built for; a Hamiltonian with another spin_size must fail
    _, negf, _ = _molecule_with_leads(negf_spin)
    H, _, _ = _molecule_with_leads(H_spin)
    E = np.array([0.])
    with pytest.raises(ValueError, match="spin_size"):
        negf.DOS(H, E, spin=0)
    with pytest.raises(ValueError, match="spin_size"):
        negf.PDOS(H, E, spin=0)


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
    # passes); the bracketed search needs at most 27 passes here, and the
    # budget leaves about 50% of margin.
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
    for Ef0 in [-5., -1., -0.5, 0., 0.2, 0.5, 0.7, 0.8, 1., 1.2, 1.5]:
        negf.Ef = Ef0
        passes[0] = 0
        try:
            ni, _ = negf.calc_n_open(H, H.q, qtol=qtol)
        except Budget:
            stalled.append(Ef0)
            continue
        assert abs(ni.sum() - H.q.sum()) <= qtol
    assert stalled == []


def test_fermi_search_far_outside_bands(fermi_path):
    # Far below the bands N(Ef) no longer changes, so no step brackets the
    # charge. The search steps up by at most 1 eV per pass and then fails
    # clearly instead of running away.
    H, _, negf, _ = _open_system(_polarized_density(), 'polarized')
    q = H.q + np.array([0.2, 0.1])

    # 20 eV below the bands is reached within the budget
    negf.Ef = -20.
    ni, _ = negf.calc_n_open(H, q, qtol=1e-6)
    assert abs(ni.sum() - q.sum()) <= 1e-6
    path = fermi_path(negf, H)
    assert np.all(np.diff(path[:12]) == pytest.approx(1., abs=1e-12))

    negf.Ef = -200.
    with pytest.raises(RuntimeError, match="brackets the charge"):
        negf.calc_n_open(H, q, qtol=1e-6)
    path = fermi_path(negf, H)
    assert len(path) == 51
    assert np.allclose(np.diff(path), 1., rtol=0, atol=1e-9)


class _TooManyPasses(Exception):
    pass


def _limit_passes(monkeypatch, negf, H, limit):
    # Make a search that does not end fail after `limit` density passes
    _G = negf_module._G
    c0 = negf.CC_eq[0, 0]
    calls = [0]

    def counter(e, *args):
        if e.imag == c0.imag:
            calls[0] += 1
            if calls[0] > limit * H.spin_size * len(H.mp.k):
                raise _TooManyPasses
        return _G(e, *args)
    monkeypatch.setattr(negf_module, '_G', counter)


def test_fermi_search_counts_nonpositive_charge(monkeypatch):
    # So far below the bands the contour gives a charge of about -1e-11.
    # That is an ordinary point with dq < 0, so the search steps up and
    # fails after its 50 steps without a bracket
    H, _, negf, _ = _open_system(_polarized_density(), 'polarized')
    q = H.q + np.array([0.2, 0.1])
    _limit_passes(monkeypatch, negf, H, 100)
    negf.Ef = -1e6
    with pytest.raises(RuntimeError, match="brackets the charge"):
        negf.calc_n_open(H, q, qtol=1e-6)


def test_fermi_search_bracket_at_resolution(monkeypatch):
    # A qtol below the precision of the charge: the bracket shrinks to
    # adjacent floats and the search fails instead of repeating Ef
    geom = sisl.Geometry([0., 0., 0.], sisl.Atom(1, R=1.), lattice=[10., 10., 10.])
    geom.set_nsc([1, 1, 1])
    TB = sisl.Hamiltonian(geom)
    H = HubbardHamiltonian(TB, U=0., kT=0.025)
    negf = NEGF(H, [sisl.WideBandSE(1, 0.1)], [[0]])
    _limit_passes(monkeypatch, negf, H, 500)
    with pytest.raises(RuntimeError, match="cannot shrink the bracket"):
        negf.calc_n_open(H, 0.123456789123, qtol=1e-20)
