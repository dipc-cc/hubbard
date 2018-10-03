from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
import sisl
import numpy as np


class Wavefunction(GeometryPlot):

    def __init__(self, HubbardHamiltonian, wf, **keywords):

        # Set default keywords
        if 'realspace' in keywords:
            if 'facecolor' not in keywords:
                keywords['facecolor'] = 'None'
            if 'cmap' not in keywords:
                keywords['cmap'] = 'Greys'
        else:
            if 'cmap' not in keywords:
                keywords['cmap'] = plt.cm.bwr

        GeometryPlot.__init__(self, HubbardHamiltonian, **keywords)

        if 'realspace' in keywords:
            self.__realspace__(wf, **keywords)

        else:
            self.__orbitals__(HubbardHamiltonian, wf, **keywords)

    def __orbitals__(self, HubbardHamiltonian, wf, **keywords):    

        x = HubbardHamiltonian.geom[:, 0]
        y = HubbardHamiltonian.geom[:, 1]

        assert len(x) == len(wf)

        self.axes.scatter(x, y, wf.real, 'r') # pos. part, marker AREA is proportional to data
        self.axes.scatter(x, y, -wf.real, 'g') # neg. part
    
    def __realspace__(self, wf, z=1.1, vmax=0.006, grid_unit=0.05, **keywords):
        
        grid, index = self.real_space_grid(wf, z, grid_unit) 

        # Create custom map to differenciate it from polarization cmap
        import matplotlib.colors as mcolors
        custom_map = mcolors.LinearSegmentedColormap.from_list(name='custom_map', colors =['g', 'white', 'red'], N=100)
        
        # Plot only the real part
        ax = self.axes.imshow(grid.grid[:, :, index[2]].T.real, cmap=custom_map, origin='lower',
                              vmax=vmax, vmin=-vmax, extent=self.extent)
        # Colorbars
        if 'colorbar' in keywords:
            if keywords['colorbar'] != False:
                # Charge density per unit of length in the z-direction
                plt.colorbar(ax)