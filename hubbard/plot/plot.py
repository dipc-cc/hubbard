import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Install font rc params
plt.rc('text', usetex=True)
plt.rc('font', family='Dejavu Sans', size=16)


class Plot(object):
    """ Class to create plot objects """

    def __init__(self, **kwargs):
        # Figure size
        figsize = kwargs.get("figsize", (8, 6))
        if "figure" in kwargs:
            self.fig = kwargs["figure"]
        else:
            self.fig = plt.figure(figsize=figsize)
        axes = self.fig.get_axes()
        if len(axes) == 0:
            self.axes = self.fig.add_subplot(1, 1, 1)
        elif len(axes) == 1:
            self.axes = axes[0]

    def savefig(self, fn):
        """ Save figure to external file

        Parameters
        ----------
        fn: str
            external file name to save plot

        See Also
        --------
        `matplotlib.pyplot.savefig`
        """
        self.fig.tight_layout()
        self.fig.savefig(fn)

    def close(self):
        """ Close figure """
        # matplotlib does not have close method on Figure class
        # weird.
        self.fig.clear()
        plt.close(self.fig)

    def set_title(self, title, fontsize=16):
        """ Set figure title

        Parameters
        ----------
        title: str
            figure title
        fontsize: int, optional
            title fontsize, defaults to 16
        """
        self.axes.set_title(title, size=fontsize)

    def set_xlabel(self, label, fontsize=16):
        """ Set label for the x-axis

        Parameters
        ----------
        label: str
            label for the x-axis
        fontsize: int, optional
            label fontsize
        """
        self.axes.set_xlabel(label, fontsize=fontsize)

    def set_ylabel(self, label, fontsize=16):
        """ Set label for the y-axis

        Parameters
        ----------
        label: str
            label for the y-axis
        fontsize: int, optional
            label fontsize
        """
        self.axes.set_ylabel(label, fontsize=fontsize)

    def set_xlim(self, xmin, xmax):
        """ Set maximum and minimum x-axis values

        Parameters
        ----------
        xmin: float
            minimum value to show in the x-axis
        xmax: float
            maximum value to show in the x-axis
        """
        self.axes.set_xlim(xmin, xmax)

    def set_ylim(self, ymin, ymax):
        """ Set maximum and minimum y-axis values

        Parameters
        ----------
        ymin: float
            minimum value to show in the y-axis
        ymax: float
            maximum value to show in the y-axis
        """
        self.axes.set_ylim(ymin, ymax)

    def add_colorbar(self, layer, pos='right', size='5%'):
        """ Add figure colorbar

        Parameters
        ----------
        layer: matplotlib.cm.ScalarMappable
            i.e., `AxesImage`, `ContourSet`, etc. described by this colorbar
        pos: str, optional
            position of the colorbar with respect to axes
        size: str, optional
            size of the colorbar in %
        """
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes(pos, size=size, pad=0.1)
        self.colorbar = plt.colorbar(layer, cax=cax)
        self.fig.subplots_adjust(right=0.8)

    def legend(self, **kwargs):
        """ Add legend to figure. It takes into account possible repeated labels and show them once """
        handles, labels = self.fig.gca().get_legend_handles_labels()
        # Reduce in case there are repeated labels
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        self.fig.legend(handles, labels, **kwargs)


class GeometryPlot(Plot):
    """ Class to create geometry plots

    Parameters
    ----------
    geometry: sisl.Geoemtry
        geometry
    ext_geom: sisl.Geometry, optional
        full sp2 geometry that may include other atoms that are typically not involved
        in a pi-tight-binding or mean-feild Hubbard calculation e.g., as Hydrogen atoms
    bdx: int, optional
        added space between geometry and figure axes
    """

    def __init__(self, geometry, ext_geom=None, bdx=2, **kwargs):

        super().__init__(**kwargs)

        self.geometry = geometry
        self.set_axes(bdx=bdx)
        # Relevant kwargs
        kw = {}
        for k in kwargs:
            if k in ['cmap']:
                kw[k] = kwargs[k]
            if k in ['facecolor']:
                kw[k] = kwargs[k]
            if k in ['label']:
                kw[k] = kwargs[k]
        # Patches
        pi = []
        aux = []
        if ext_geom:
            g = ext_geom
        else:
            g = self.geometry
        for ia in g:
            idx = g.close(ia, R=[0.1, 1.6])
            if g.atoms[ia].Z == 1: # H
                aux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.4))
            elif g.atoms[ia].Z == 5: # B
                self.axes.add_patch(patches.Circle((np.average(g.xyz[ia, 0]), np.average(g.xyz[ia, 1])), radius=0.7, color='r', lw=2, fill=False))
                pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
            elif g.atoms[ia].Z == 6: # C
                if len(idx[1]) == 4:
                    # If the C atom has 4 neighbours (sp3 configuration) it will be represented
                    # as an aux site
                    aux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
                    # Add a blue patch at the H positions
                    Hsp3 = [i for i in idx[1] if g.atoms[i].Z == 1]
                    self.axes.add_patch(patches.Circle((np.average(g.xyz[Hsp3, 0]), np.average(g.xyz[Hsp3, 1])), radius=1.4, alpha=0.15, fc='c'))
                else:
                    pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
            elif g.atoms[ia].Z == 7: # N
                self.axes.add_patch(patches.Circle((np.average(g.xyz[ia, 0]), np.average(g.xyz[ia, 1])), radius=0.7, color='g', lw=2, fill=False))
                pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
            elif g.atoms[ia].Z > 10: # Some other atom
                aux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.2))

        # Pi sites
        ppi = PatchCollection(pi, alpha=1., lw=1.2, edgecolor='0.6', **kw)
        ppi.set_array(np.zeros(len(pi)))
        ppi.set_clim(-1, 1)
        self.ppi = ppi
        self.axes.add_collection(self.ppi)
        # Aux sites
        paux = PatchCollection(aux, alpha=1., lw=1.2, edgecolor='0.6', **kw)
        paux.set_array(np.zeros(len(aux)))
        paux.set_clim(-1, 1)
        self.paux = paux
        self.axes.add_collection(self.paux)

    def __orbitals__(self, v, **kwargs):
        # Set values for the pi-network
        self.ppi.set_array(v)

        # Set color range
        if 'vmax' in kwargs:
            if 'vmin' in kwargs:
                vmin = kwargs['vmin']
            else:
                vmin = -kwargs['vmax']
            self.ppi.set_clim(vmin, kwargs['vmax'])
        else:
            self.ppi.set_clim(min(v), max(v))

        # Colorbars
        if 'colorbar' in kwargs:
            if kwargs['colorbar'] != False:
                self.add_colorbar(self.ppi)
                if 'label' in kwargs:
                    self.set_colorbar_ylabel(kwargs['label'])

    def __realspace__(self, v, z=1.1, grid_unit=[100, 100, 1], density=False, smooth=False, **kwargs):

        def real_space_grid(v, grid_unit, density):
            import sisl

            # Create a temporary copy of the geometry
            g = self.geometry.copy()

            # Set new sc to create real-space grid
            sc = sisl.SuperCell([self.xmax-self.xmin, self.ymax-self.ymin, 1000], origo=[0, 0, -z])
            g.set_sc(sc)

            # Move geometry within the supercell
            g = g.move([-self.xmin, -self.ymin, -np.amin(g.xyz[:, 2])])
            # Make z~0 -> z = 0
            g.xyz[np.where(np.abs(g.xyz[:, 2]) < 1e-3), 2] = 0

            # Create the real-space grid
            grid = sisl.Grid(grid_unit, sc=g.sc, geometry=g)

            if density:
                D = sisl.physics.DensityMatrix(g)
                a = np.arange(len(D))
                D.D[a, a] = v
                D.density(grid)
            else:
                if isinstance(v, sisl.physics.electron.EigenstateElectron):
                    # Set parent geometry equal to the temporary one
                    v.parent = g
                    v.wavefunction(grid)
                else:
                    # In case v is a vector
                    sisl.electron.wavefunction(v, grid, geometry=g)
            del g
            return grid

        grid = real_space_grid(v, grid_unit, density)
        if smooth:
            # Smooth grid with gaussian function
            if 'r_smooth' in kwargs:
                r_smooth = kwargs['r_smooth']
            else:
                r_smooth = 0.7
            grid = grid.smooth(method='gaussian', r=r_smooth)

        slice_grid = grid.grid[:, :, 0].T.real

        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
        else:
            vmax = np.amax(np.absolute(slice_grid))

        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
        else:
            vmin = -vmax

        # Plot only the real part of the grid
        # The image will be created in an imshow layer (stored in self.imshow)
        self.imshow = self.axes.imshow(slice_grid, cmap='seismic', origin='lower',
                              vmax=vmax, vmin=vmin, extent=self.extent)
        # Colorbars
        if 'colorbar' in kwargs:
            if kwargs['colorbar'] != False:
                self.add_colorbar(self.imshow)
                if 'label' in kwargs:
                    self.set_colorbar_ylabel(kwargs['label'])

    def set_axes(self, bdx=2):
        g = self.geometry
        x = g.xyz[:, 0]
        y = g.xyz[:, 1]
        self.xmin = min(x)-bdx
        self.xmax = max(x)+bdx
        self.ymin = min(y)-bdx
        self.ymax = max(y)+bdx
        self.extent = [min(x)-bdx, max(x)+bdx, min(y)-bdx, max(y)+bdx]
        self.set_xlim(self.xmin, self.xmax)
        self.set_ylim(self.ymin, self.ymax)
        self.set_xlabel(r'$x$ (\AA)')
        self.set_ylabel(r'$y$ (\AA)')
        self.axes.set_aspect('equal')

    def set_colorbar_ylabel(self, label, fontsize=20):
        self.colorbar.ax.set_ylabel(label, fontsize=fontsize)

    def set_colorbar_yticks(self, ticks):
        self.colorbar.ax.set_yticks(ticks)

    def get_colorbar_yticks(self):
        return self.colorbar.ax.get_yticks()

    def set_colorbar_yticklabels(self, labels=None, fontsize=20):
        if not labels:
            labels = self.colorbar.ax.get_yticks()
        self.colorbar.ax.set_yticklabels(labels, fontsize=fontsize)

    def set_colorbar_xlabel(self, label, fontsize=20):
        self.colorbar.ax.set_xlabel(label, fontsize=fontsize)

    def get_colorbar_xticks(self):
        return self.colorbar.ax.get_xticks()

    def set_colorbar_xticks(self, ticks):
        self.colorbar.ax.set_xticks(ticks)

    def set_colorbar_xticklabels(self, labels=None, fontsize=20):
        if labels != None:
            labels = self.colorbar.ax.get_xticks()
        self.colorbar.ax.set_xticklabels(labels, fontsize=fontsize)

    def annotate(self, sites=[], size=6):
        """ Annotate the site indices in the pi-network

        sites: array_like, optional
            specify sites to be annotated in figure
        size: int, optional
            font size for the annotation
        """
        g = self.geometry
        x = g.xyz[:, 0]
        y = g.xyz[:, 1]
        if not sites:
            sites = g
        for ia in sites:
            self.axes.annotate(ia, (x[ia], y[ia]), size=size)
