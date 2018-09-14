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

        # Colorbars
        self.ppi.set_clim(min(charge), max(charge))
        self.paux.set_clim(-1, 1)
        if colorbar:
            self.add_colorbar(self.ppi, label=r'$Q_\uparrow+Q_\downarrow$ ($e$)')

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

        # Colorbars
        cmax = max(abs(chgdiff))
        self.ppi.set_clim(-cmax, cmax)
        self.paux.set_clim(-1, 1)
        if colorbar:
            self.add_colorbar(self.ppi, label=r'$Q_\uparrow+Q_\downarrow-Q_\mathrm{N.A.}$ ($e$)')

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

        # Colorbars
        cmax = max(abs(spinpol))
        self.ppi.set_clim(-cmax, cmax)
        self.paux.set_clim(-1, 1)
        if colorbar:
            self.add_colorbar(self.ppi, label=r'$Q_\uparrow-Q_\downarrow$ ($e$)')

        # Write file
        fn = HubbardHamiltonian.get_label()+'-pol.pdf'
        self.savefig(fn)

        # Close plot
        self.close()


import sisl
import numpy as np

class SpinPolarizationRS(GeometryPlot):

    def __init__(self, HubbardHamiltonian, z=1.1, vmax=0.006, grid_unit=0.05, colorbar=True):

        GeometryPlot.__init__(self, HubbardHamiltonian, cmap='Greys', facecolor='None')

        # These limits ensure that the edgecolor for the patches stay the same
        # grey ones as for other plots (patch arrays are set to zero)
        self.ppi.set_clim(-1, 1)
        self.paux.set_clim(-1, 1)

        # As the radial extension is only 1.6 ang, two times this should
        # be enough for the supercell in the z-direction:
        sc = sisl.SuperCell([self.xmax-self.xmin, self.ymax-self.ymin, 3.2])
        grid = sisl.Grid(grid_unit, sc=sc)

        # The following is a bit of black magic...
        # not sure this gives the density on the grid  
        vecs = np.zeros((HubbardHamiltonian.sites, HubbardHamiltonian.sites))
        vecs[0, :] = HubbardHamiltonian.nup - HubbardHamiltonian.ndn
        H = HubbardHamiltionian.H0.move([self.xmax, self.ymax, 0])
        H.set_sc(sc)
        es = sisl.EigenstateElectron(vecs, np.zeros(HubbardHamiltonian.sites), H)
        es.sub(0).psi(grid)
        index = grid.index([0, 0, z])

        # Plot only the real part
        ax = self.axes.imshow(grid.grid[:, :, index[2]].T.real, cmap='seismic', origin='lower',
                              vmax=vmax, vmin=-vmax, extent=self.extent)
        if colorbar:
            self.add_colorbar(ax, label=r'$Q_\uparrow-Q_\downarrow$ ($e$)')

        # Write file
        fn = HubbardHamiltonian.get_label()+'-rs-pol.pdf'
        self.savefig(fn)

        # Close plot
        self.close()
