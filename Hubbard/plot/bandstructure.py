from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import Plot
import numpy as np

__all__ = ['Bandstructure']


class Bandstructure(Plot):
    """ Plot the bandstructure for the `HubbardHamiltonian` object of a 1D system """

    def __init__(self, HubbardHamiltonian, ymax=4., projection=None, spin=0, scale=1, c='r', **keywords):

        # Set default keywords
        if 'figsize' not in keywords:
            keywords['figsize'] = (4, 8)

        Plot.__init__(self, **keywords)

        self.set_xlabel(r'$ka/\pi$')
        self.set_ylabel(r'$E_{nk}-E_\mathrm{mid}$ (eV)')
        self.set_ylim(-ymax, ymax)

        self.add_bands(HubbardHamiltonian, projection=projection, spin=spin, scale=scale, c=c)

    def add_bands(self, HubbardHamiltonian, projection=None, spin=0, scale=1, c='r'):
        # Get TB bands
        ka = HubbardHamiltonian.mp.k
        ev = np.empty([len(ka), 2, HubbardHamiltonian.sites])
        pdos = np.empty([len(ka), 2, HubbardHamiltonian.sites])
        # Loop over k
        for ik, k in enumerate(ka):
            for ispin in range(2):
                ev[ik, ispin], evec = HubbardHamiltonian.eigh(k, eigvals_only=False, spin=ispin)
                if projection != None:
                    v = evec[projection]
                    pdos[ik, ispin] = np.diagonal(np.dot(np.conjugate(v).T, v).real)
        # Set energy reference to midgap
        HubbardHamiltonian.find_midgap()
        ev -= HubbardHamiltonian.midgap
        # Make plot
        x = 2*ka # Units ka/pi
        if not np.allclose(ev[:, 0], ev[:, 1]):
            # Add spin-down component to plot
            plt.plot(x, ev[:, 1], 'g.')
        # Fat bands?
        if projection != None:
            for i, evi in enumerate(ev[0, 0]):
                plt.errorbar(x, ev[:, 0, i], yerr=scale*pdos[:, 0, i], alpha=.4, color='Grey')
        # Add spin-up component to plot (top layer)
        x = np.linspace(0, 0.5, len(ka))
        plt.plot(x, ev[:, 0], color=c)
        # Adjust borders
        plt.subplots_adjust(left=0.2, top=.95, bottom=0.1, right=0.95)
