import matplotlib.pyplot as plt
from hubbard.plot import GeometryPlot
import sisl
import numpy as np
from hubbard.grid import *

__all__ = ['Wavefunction']


class Wavefunction(GeometryPlot):
    """ Plot the wavefunction with its complex phase for the `hubbard.HubbardHamiltonian` object

    Parameters
    ----------
    HubbardHamiltonian: hubbard.hamiltonian.HubbardHamiltonian instance
        hubbard.hamiltonian object of a system
    wf: array_like
        vector that contains the eigenstate to plot
    ext_geom: sisl.Geometry instance
        external geometry to plot the GeometryPlot behind the wavefunction layer
        if no ext_geom is passed, it uses the geometry stored in the `hubbard.HubbardHamiltonian` object

    Notes
    -----
    If `realspace` kwarg is passed it plots the wavefunction in a realspace grid
    In this case either a `sisl.SuperCell` (`sc` kwarg) or the `z` kwarg to slice the real space grid at the desired z coordinate needs to be passed
    In other case the wavefunction is plotted as a scatter plot, where the size of the blobs depend on the value
    of the coefficient of `wf` on the atomic sites
    """

    def __init__(self, HH, wf, ext_geom=None, cb_label=r'Phase', realspace=False, **kwargs):

        # Set default kwargs
        if realspace:
            if 'facecolor' not in kwargs:
                kwargs['facecolor'] = 'None'
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'Greys'
        else:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.bwr

        super().__init__(HH.geometry, ext_geom=ext_geom, **kwargs)

        self.plot_wf(HH, wf, cb_label=cb_label, realspace=realspace, **kwargs)

    def plot_wf(self, HH, wf, cb_label=r'Phase', realspace=False, **kwargs):
        if realspace:
            if 'shape' not in kwargs:
                kwargs['shape'] = [100,100,1]

            if 'vmin' not in kwargs:
                kwargs['vmin'] = 0

            xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

            if 'sc' not in kwargs:
                if 'z' in kwargs:
                    origin = [xmin, ymin, -kwargs['z']]
                else:
                    raise ValueError('Either a SC or the z coordinate to slice the real space grid needs to be passed')

                kwargs['sc'] = sisl.SuperCell([xmax-xmin, ymax-ymin, 1000], origin=origin)

            if 'axis' not in kwargs:
                kwargs['axis'] = 2

            grid = real_space_grid(self.geometry, kwargs['sc'], wf, kwargs['shape'], mode='wavefunction')

            # Slice it to obtain a 2D grid
            slice_grid = grid.swapaxes(kwargs['axis'], 2).grid[:, :, 0].T.real

            self.__realspace__(slice_grid, **kwargs)

            # Create custom map to differenciate it from polarization cmap
            import matplotlib.colors as mcolors
            custom_map = mcolors.LinearSegmentedColormap.from_list(name='custom_map', colors =['g', 'white', 'red'], N=100)
            self.imshow.set_cmap(custom_map)

        else:
            x = HH.geometry[:, 0]
            y = HH.geometry[:, 1]

            assert len(x) == len(wf)

            fun, phase = np.absolute(wf), np.angle(wf)

            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list(name='custom_map', colors =['red', 'yellow', 'green', 'blue', 'red'], N=500)

            ax = self.axes.scatter(x, y, s=fun, c=phase, cmap=cmap, vmax=np.pi, vmin=-np.pi)

            # Colorbars
            if 'colorbar' in kwargs:
                if kwargs['colorbar'] != False:
                    cb = self.fig.colorbar(ax, label=cb_label)
                    if 'ticks' in kwargs:
                        cb.set_ticks([np.pi, 3*np.pi/4, np.pi/2, np.pi/4, 0, -np.pi/4., -np.pi/2., -3*np.pi/4, -np.pi, ])
                        cb.set_ticklabels([r'$\pi$', r'$3\pi/4$', r'$\pi/2$', r'$\pi/4$', r'$0$', r'-$\pi/4$', r'-$\pi/2$', r'-$3\pi/4$', r'-$\pi$'])
