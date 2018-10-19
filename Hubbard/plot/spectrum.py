from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import Plot
import numpy as np


class Spectrum(Plot):

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], xmax=10, ymax=0, **keywords):

        Plot.__init__(self, **keywords)
        self.axes.fill_between([-xmax, 0], 0, 1.0, facecolor='k', alpha=0.1)
        lmax = 0.0
        for ispin in range(2):
            ev, L = HubbardHamiltonian.calc_orbital_charge_overlaps(k=k, spin=ispin)
            ev -= HubbardHamiltonian.midgap
            L = np.diagonal(L)
            lmax = max(max(L), lmax)
            plt.plot(ev, L, 'rg'[ispin]+'.+'[ispin], label=[r'$\sigma=\uparrow$', r'$\sigma=\downarrow$'][ispin])
            for i in range(len(ev)):
                self.axes.annotate(i, (ev[i], L[i]), fontsize=6)
        self.axes.legend()
        self.set_xlabel(r'$E_{\alpha\sigma}-E_\mathrm{mid}$ (eV)')
        self.set_ylabel(r'$\eta_{\alpha\sigma}=\int dr |\psi_{\alpha\sigma}|^4$')
        self.set_xlim(-xmax, xmax)
        if ymax == 0:
            self.set_ylim(0, lmax)
        else:
            self.set_ylim(0, ymax)


class LDOSmap(Plot):

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], spin=0, axis=0,
                 nx=501, gamma_x=1.0, dx=5.0, ny=501, gamma_e=0.05, ymax=10.,
                 **keywords):

        Plot.__init__(self, **keywords)
        ev, evec = HubbardHamiltonian.eigh(k=k, eigvals_only=False, spin=spin)
        ev -= HubbardHamiltonian.midgap
        coord = HubbardHamiltonian.geom.xyz[:, axis]

        xmin, xmax = min(coord)-dx, max(coord)+dx
        ymin, ymax = -ymax, ymax
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        dat = np.zeros((len(x), len(y)))
        for i, evi in enumerate(ev):
            de = gamma_e/((y-evi)**2+gamma_e**2)
            dos = np.zeros(len(x))
            for i, vi in enumerate(evec[:, i]):
                dos += vi**2*gamma_x/((x-coord[i])**2+gamma_x**2)
            dat += np.outer(dos, de)

        cm = plt.cm.hot
        self.axes.imshow(dat.T, extent=[xmin, xmax, ymin, ymax], cmap=cm, origin='lower')
        self.set_xlabel(r'$x$ (\AA)')
        self.set_ylabel(r'$E-E_\mathrm{midgap}$ (eV)')
        self.set_xlim(xmin, xmax)
        self.set_ylim(ymin, ymax)
        self.axes.set_aspect('auto')
