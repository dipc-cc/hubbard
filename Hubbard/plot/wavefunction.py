from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
import sisl
import numpy as np


class Wavefunction(GeometryPlot):

    def __init__(self, HubbardHamiltonian, wf, label=r'Phase', **keywords):

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

            # Create custom map to differenciate it from polarization cmap
            import matplotlib.colors as mcolors
            custom_map = mcolors.LinearSegmentedColormap.from_list(name='custom_map', colors =['g', 'white', 'red'], N=100)
            self.imshow.set_cmap(custom_map)

        else:
            x = HubbardHamiltonian.geom[:, 0]
            y = HubbardHamiltonian.geom[:, 1]

            assert len(x) == len(wf)

            fun, phase = np.absolute(wf), np.angle(wf)
            # Make phases -pi -> pi
            phase[np.where(phase+np.pi < 1e-5)] = np.pi 

            import matplotlib.colors as mcolors
            custom_map = mcolors.LinearSegmentedColormap.from_list(name='custom_map', colors =['b', 'red'], N=100)
            ax = self.axes.scatter(x, y, s=fun, c=phase, cmap=custom_map)

            # Colorbars
            if 'colorbar' in keywords:
                if keywords['colorbar'] != False:
                    cb = self.fig.colorbar(ax, label=label)
                    if 'ticks' in keywords:
                        cb.set_ticks([np.pi, 3*np.pi/4, np.pi/2, np.pi/4, 0, -np.pi/4., -np.pi/2., -3*np.pi/4, -np.pi,])
                        cb.set_ticklabels([r'$\pi$', r'$3\pi/4$', r'$\pi/2$', r'$\pi/4$', r'$0$', r'-$\pi/4$', r'-$\pi/2$', r'-$3\pi/4$', r'-$\pi$'])

