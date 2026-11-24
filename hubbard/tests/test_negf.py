import numpy as np
import pytest
import sisl

import hubbard.density as density
from hubbard import HubbardHamiltonian, sp2
from hubbard.negf import NEGF

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


class _WideBandOverride(sisl.WideBandSE):
    # A user's wide-band electrode with its own, energy-independent coupling
    def self_energy(self, *args, **kwargs):
        return np.diag(np.repeat(-0.4j, len(self)))


@pytest.mark.parametrize("cls, coupling", [
    (sisl.WideBandSE, 0.1),
    (_WideBandNoOverride, 0.1),
    (_WideBandOverride, 0.4),
], ids=["WideBandSE", "subclass", "subclass-override"])
def test_wideband_coupling(cls, coupling):
    # A WideBandSE(n, eta) couples its orbitals with -i*eta*I at every energy,
    # whatever the installed sisl version; a subclass that overrides
    # self_energy couples with what its own method returns
    eta = 0.1
    H = _molecule(kT_contour)
    no = len(H.H)
    negf = NEGF(H, [cls(no, eta)], [np.arange(no)])

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
