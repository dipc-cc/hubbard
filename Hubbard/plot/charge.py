from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot

class Charge(GeometryPlot):

    def __init__(self, HubbardHamiltonian, colorbar=True):

        GeometryPlot.__init__(self, HubbardHamiltonian, cmap=plt.cm.bwr)

        # Compute total charge on each site
        charge = HubbardHamiltonian.nup + HubbardHamiltonian.ndn

        # Set values for the pi-network
        self.ppi.set_array(charge)

        # Colorbar
        self.ppi.set_clim(min(charge), max(charge))
        if colorbar:
            self.add_colorbar(label=r'$Q_\uparrow+Q_\downarrow$ ($e$)')

        # Write file
        fn = HubbardHamiltonian.get_label()+'-chg.pdf'
        self.savefig(fn)

        # Close plot
        self.close()


class ChargeDifference(GeometryPlot):

    def __init__(self, HubbardHamiltonian, colorbar=True):

        GeometryPlot.__init__(self, HubbardHamiltonian, cmap=plt.cm.bwr)

        # Compute charge difference to neutral atom on each site
        chgdiff = HubbardHamiltonian.nup + HubbardHamiltonian.ndn
        for ia in HubbardHamiltonian.pi_geom:
            chgdiff[ia] -= HubbardHamiltonian.pi_geom.atoms[ia].Z-5

        # Set values for the pi-network
        self.ppi.set_array(chgdiff)

        # Colorbar
        cmax = max(abs(chgdiff))
        self.ppi.set_clim(-cmax, cmax)
        if colorbar:
            self.add_colorbar(label=r'$Q_\uparrow+Q_\downarrow-Q_\mathrm{N.A.}$ ($e$)')

        # Write file
        fn = HubbardHamiltonian.get_label()+'-chgdiff.pdf'
        self.savefig(fn)

        # Close plot
        self.close()

class SpinPolarization(GeometryPlot):

    def __init__(self, HubbardHamiltonian, colorbar=True):

        GeometryPlot.__init__(self, HubbardHamiltonian, cmap=plt.cm.bwr)

        # Compute charge difference to neutral atom on each site
        spinpol = HubbardHamiltonian.nup - HubbardHamiltonian.ndn

        # Set values for the pi-network
        self.ppi.set_array(spinpol)

        # Colorbar
        cmax = max(abs(spinpol))
        self.ppi.set_clim(-cmax, cmax)
        if colorbar:
            self.add_colorbar(label=r'$Q_\uparrow-Q_\downarrow$ ($e$)')

        # Write file
        fn = HubbardHamiltonian.get_label()+'-pol.pdf'
        self.savefig(fn)

        # Close plot
        self.close()
