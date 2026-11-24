import numpy as np
import pytest
import sisl

import hubbard.density as density
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
