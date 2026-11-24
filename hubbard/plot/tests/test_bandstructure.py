import matplotlib
matplotlib.use("Agg")
import numpy as np
import sisl

from hubbard import HubbardHamiltonian, sp2, density, plot


def _ribbon():
    # Antiferromagnetic zigzag ribbon, spin channels differ
    g = sisl.geom.zgnr(2)
    H = HubbardHamiltonian(sp2(g), U=3., nkpt=[10, 1, 1])
    H.set_polarization([0], dn=[len(g) - 1])
    H.converge(density.calc_n, tol=1e-8)
    return H


def test_fat_bands_spin1():
    R = _ribbon()
    projection = [0]
    scale = 1
    p = plot.Bandstructure(R, projection=projection, spin=1, scale=scale)
    try:
        # Reference: weight of the projected sites in each spin-1 eigenvector,
        # from a direct numpy diagonalization along the default Gamma-X path
        band = sisl.BandStructure(R.H, [[0., 0., 0.], [0.5, 0., 0.]], 101)
        ev = np.empty([len(band.k), R.sites])
        w = np.empty([len(band.k), R.sites])
        for ik, k in enumerate(band.k):
            e, c = np.linalg.eigh(R.H.Hk(k, spin=1, format="array"))
            ev[ik] = e
            w[ik] = (np.abs(c[projection, :]) ** 2).sum(axis=0)
        # The spin-0 projection differs, so a spin mix-up would show
        c0 = np.linalg.eigh(R.H.Hk(band.k[10], spin=0, format="array"))[1]
        assert not np.allclose((np.abs(c0[projection, :]) ** 2).sum(axis=0), w[10])

        bars = p.axes.containers
        assert len(bars) == R.sites
        for i, bar in enumerate(bars):
            data_line, _, (barcol,) = bar.lines
            assert np.allclose(data_line.get_ydata(), ev[:, i])
            seg = np.array(barcol.get_segments())
            yerr = (seg[:, 1, 1] - seg[:, 0, 1]) / 2
            assert np.allclose(yerr, scale * w[:, i], atol=1e-10)
    finally:
        p.close()
