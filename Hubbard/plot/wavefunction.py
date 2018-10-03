from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
import sisl
import numpy as np


class Wavefunction(GeometryPlot):

    def __init__(self, HubbardHamiltonian, wf, **keywords):

        # Set default keywords
        if 'cmap' not in keywords:
            keywords['cmap'] = plt.cm.bwr

        GeometryPlot.__init__(self, HubbardHamiltonian, **keywords)

        if 'realspace' in keywords:
            self.__realspace__(HubbardHamiltonian, wf, **keywords)

        else:
            self.__orbitals__(HubbardHamiltonian, wf, **keywords)

    def __orbitals__(self, HubbardHamiltonian, wf, **keywords):    

        x = HubbardHamiltonian.geom[:, 0]
        y = HubbardHamiltonian.geom[:, 1]

        assert len(x) == len(wf)

        self.axes.scatter(x, y, wf.real, 'r') # pos. part, marker AREA is proportional to data
        self.axes.scatter(x, y, -wf.real, 'g') # neg. part
    
    def __realspace__(self, HubbardHamiltonian, wf, z=1.1, vmax=0.006, grid_unit=0.05, **keywords):
        sc = sisl.SuperCell([self.xmax-self.xmin, self.ymax-self.ymin, 3.2])
        grid = sisl.Grid(grid_unit, sc=sc)
        
        vecs = np.zeros((HubbardHamiltonian.sites, HubbardHamiltonian.sites))
        vecs[0, :] = wf
        H = HubbardHamiltonian.H.move([-self.xmin, -self.ymin, 0])
        H.xyz[np.where(np.abs(H.xyz[:, 2]) < 1e-3), 2] = 0
        H.set_sc(sc)
        es = sisl.EigenstateElectron(vecs, np.zeros(HubbardHamiltonian.sites), H)
        es.sub(0).psi(grid)
        index = grid.index([0, 0, z])

        # Plot only the real part
        ax = self.axes.imshow(grid.grid[:, :, index[2]].T.real, cmap='seismic', origin='lower',
                              vmax=vmax, vmin=-vmax, extent=self.extent)
        # Colorbars
        if 'colorbar' in keywords:
            if keywords['colorbar'] != False:
                # Charge density per unit of length in the z-direction
                plt.colorbar(ax, label=r'$q_\uparrow+q_\downarrow$ ($e/$\AA)')