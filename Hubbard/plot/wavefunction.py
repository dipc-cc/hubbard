from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
import sisl
import numpy as np

__all__ = ['Wavefunction']


class Wavefunction(GeometryPlot):
    """ Plot the wavefunction with its complex phase for the `HubbardHamiltonian` object

    Parameters
    ----------
    HubbardHamiltonian: Hubbard.hamiltonian.HubbardHamiltonian instance
        Hubbard.hamiltonian object of a system
    wf: array_like
        vector that contains the eigenstate to plot
    ext_geom: sisl.Geometry instance
        external geometry to plot the GeometryPlot behind the wavefunction layer
        if no ext_geom is passed, it uses the geometry stored in the HubbardHamiltonian object

    Notes
    -----
    If `realspace` keyword is passed it plots the wavefunction in a realspace grid
    In other case the wavefunction is plotted as a scatter plot, where the size of the blobs depend on the value
    of the coefficient of `wf` on the atomic sites
    """

    def __init__(self, HubbardHamiltonian, wf,  ext_geom=None, cb_label=r'Phase', **keywords):

        # Set default keywords
        if 'realspace' in keywords:
            if 'facecolor' not in keywords:
                keywords['facecolor'] = 'None'
            if 'cmap' not in keywords:
                keywords['cmap'] = 'Greys'
        else:
            if 'cmap' not in keywords:
                keywords['cmap'] = plt.cm.bwr

        GeometryPlot.__init__(self, HubbardHamiltonian.geom, ext_geom=ext_geom, **keywords)

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

            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list(name='custom_map', colors =['red', 'yellow', 'green', 'blue', 'red'], N=500)

            ax = self.axes.scatter(x, y, s=fun, c=phase, cmap=cmap, vmax=np.pi, vmin=-np.pi)

            # Colorbars
            if 'colorbar' in keywords:
                if keywords['colorbar'] != False:
                    cb = self.fig.colorbar(ax, label=cb_label)
                    if 'ticks' in keywords:
                        cb.set_ticks([np.pi, 3*np.pi/4, np.pi/2, np.pi/4, 0, -np.pi/4., -np.pi/2., -3*np.pi/4, -np.pi, ])
                        cb.set_ticklabels([r'$\pi$', r'$3\pi/4$', r'$\pi/2$', r'$\pi/4$', r'$0$', r'-$\pi/4$', r'-$\pi/2$', r'-$3\pi/4$', r'-$\pi$'])
