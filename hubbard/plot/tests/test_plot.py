import matplotlib
matplotlib.use("Agg")
import numpy as np
import sisl

from hubbard import HubbardHamiltonian, sp2, plot


def _plot_with_colorbar():
    g = sisl.geom.zgnr(2).tile(2, 0)
    g.set_nsc([1, 1, 1])
    H = HubbardHamiltonian(sp2(g), U=3.5)
    H.set_polarization([0], dn=[len(g) - 1])
    p = plot.SpinPolarization(H, colorbar=True)
    p.set_colorbar_xticks([0.25, 0.75])
    return p


def _xlabels(p):
    return [t.get_text() for t in p.colorbar.ax.get_xticklabels()]


def test_colorbar_xticklabels_default_to_ticks():
    p = _plot_with_colorbar()
    try:
        p.set_colorbar_xticklabels()
        assert _xlabels(p) == [str(t) for t in [0.25, 0.75]]
    finally:
        p.close()


def test_colorbar_xticklabels_given():
    p = _plot_with_colorbar()
    try:
        p.set_colorbar_xticklabels(['a', 'b'])
        assert _xlabels(p) == ['a', 'b']
        assert np.allclose(p.get_colorbar_xticks(), [0.25, 0.75])
    finally:
        p.close()
