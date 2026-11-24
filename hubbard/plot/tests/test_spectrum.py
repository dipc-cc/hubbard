import matplotlib
matplotlib.use("Agg")
import numpy as np
import sisl
from matplotlib.collections import PathCollection

from hubbard import HubbardHamiltonian, sp2, density, plot


def _molecule():
    # Small zigzag flake (12 sites) in its triplet state, so the spins differ
    g = sisl.geom.zgnr(2).tile(3, 0)
    g.set_nsc([1, 1, 1])
    H = HubbardHamiltonian(sp2(g), U=3.5, q=(7, 5))
    H.set_polarization([0], dn=[len(g) - 1])
    H.converge(density.calc_n_insulator, tol=1e-8)
    return H


def test_ldos_scatter_is_lorentzian_pdos_per_site():
    # Reference: Lorentzian-broadened local DOS per site, summed over spin,
    # from a direct numpy diagonalization of each spin Hamiltonian
    H = _molecule()
    eta = 0.05
    ev0 = np.linalg.eigvalsh(H.H.Hk(spin=0, format="array"))
    E = ev0[len(ev0) // 2] + 0.02
    ref = 0
    for s in range(2):
        e, c = np.linalg.eigh(H.H.Hk(spin=s, format="array"))
        L = (eta / np.pi) / ((E - e) ** 2 + eta ** 2)
        ref = ref + np.abs(c) ** 2 @ L
    # Spin channels differ, so a mix-up would show
    assert not np.allclose(H.H.Hk(spin=0, format="array"), H.H.Hk(spin=1, format="array"))

    p = plot.LDOS(H, E, eta=eta)
    try:
        scat = [c for c in p.axes.collections if isinstance(c, PathCollection)]
        assert len(scat) == 1
        sizes = scat[0].get_sizes()
        assert sizes.shape == (H.geometry.no,)
        assert np.allclose(sizes, ref, rtol=1e-10, atol=0)
        assert np.allclose(sizes, H.PDOS([E], eta=eta)[0, :, 0], rtol=1e-10, atol=0)
    finally:
        p.close()
