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
            
