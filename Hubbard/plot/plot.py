"""

:mod:`Hubbard.plot.plot`
========================

Base classes for plots

.. currentmodule:: Hubbard.plot.plot

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plot(object):

    def __init__(self, **keywords):
        # Figure size
        if 'figsize' in keywords:
            self.fig = plt.figure(figsize=keywords['figsize'])
        else:
            self.fig = plt.figure(figsize=(8, 6))
        self.axes = plt.axes()
        plt.rc('text', usetex=True)
        plt.rc('font', family='Bitstream Vera Serif', size=16)

    def savefig(self, fn):
        self.fig.savefig(fn)
        print('Wrote', fn)

    def close(self):
        plt.close('all')

    def set_title(self, title, size=16):
        self.axes.set_title(title, size=size)

    def set_xlabel(self, label):
        self.axes.set_xlabel(label)

    def set_ylabel(self, label):
        self.axes.set_ylabel(label)

    def set_xlim(self, xmin, xmax):
        self.axes.set_xlim(xmin, xmax)

    def set_ylim(self, ymin, ymax):
        self.axes.set_ylim(ymin, ymax)

# Generate a dummy plot, this seems to avoid font issues with subsequent instances
Plot()


class GeometryPlot(Plot):

    def __init__(self, HubbardHamiltonian, **keywords):

        Plot.__init__(self, **keywords)

        self.HH = HubbardHamiltonian
        self.set_axes()
        # Relevant keywords
        kw = {}
        for k in keywords:
            if k in ['cmap']:
                kw[k] = keywords[k]
            if k in ['facecolor']:
                kw[k] = keywords[k]
            if k in ['label']:
                kw[k] = keywords[k]
        # Patches
        pi = []
        aux = []
        g = self.HH.ext_geom
        for ia in g:
            idx = g.close(ia, R=[0.1, 1.6])
            if g.atoms[ia].Z == 1: # H
                if len(idx[1]) == 2:
                    # If the H atom has 2 neighbours (sp3 conf.) it adds a blue circle at its position
                    self.axes.add_patch(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.4, alpha=0.15, fc='lightblue'))
                aux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.4))
            elif g.atoms[ia].Z == 5: # B
                pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.0))
            elif g.atoms[ia].Z == 6: # C
                if len(idx[1]) == 4:
                    # If the C atom has 4 neighbours (sp3 configuration) it will be represented 
                    # as an aux site 
                    aux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
                else:
                    pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
            elif g.atoms[ia].Z == 7: # N
                pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.0))
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
        
    def __orbitals__(self, v, label=None, **keywords):
        # Set values for the pi-network
        self.ppi.set_array(v)

        # Set color range
        if 'vmax' in keywords:
            if 'vmin' in keywords:
                vmin = keywords['vmin']
            else:
                vmin = -keywords['vmax']
            self.ppi.set_clim(vmin, keywords['vmax'])
        else:
            self.ppi.set_clim(min(v), max(v))

        # Colorbars
        if 'colorbar' in keywords:
            if keywords['colorbar'] != False:
                self.add_colorbar(self.ppi, label=label)

    def __realspace__(self, v, z=1.1, vmax=0.00006, grid_unit=0.05, density=False, label=None, **keywords):

        def real_space_grid(v, grid_unit, density):
            import sisl

            # Create a temporary copy of the geometry 
            H = self.HH.geom.copy()
            
            # Set new sc to create real-space grid
            sc = sisl.SuperCell([self.xmax-self.xmin, self.ymax-self.ymin, 3.2], origo=[self.xmin, self.ymin, 0])
            H.set_sc(sc)
        
            # Shift negative xy coordinates within the supercell
            ix = np.where(H.xyz[:, 0] < 0)
            iy = np.where(H.xyz[:, 1] < 0)
            H.xyz[ix, 0] = H.axyz(isc=[1, 0, 0])[ix, 0]
            H.xyz[iy, 1] = H.axyz(isc=[0, 1, 0])[iy, 1]
            # Make z~0 -> z = 0                      
            H.xyz[np.where(np.abs(H.xyz[:, 2]) < 1e-3), 2] = 0

            # Create the real-space grid
            grid = sisl.Grid(grid_unit, sc=H.sc, geometry=H)

            if density:
                D = sisl.physics.DensityMatrix(H)
                for ia in H:
                    D.D[ia, ia] = v[ia]
                D.density(grid)
            else:
                if isinstance(v, sisl.physics.electron.EigenstateElectron):
                    # Set parent geometry equal to the temporary one
                    v.parent = H
                    v.wavefunction(grid)
                else:
                    # In case v is a vector
                    sisl.electron.wavefunction(v, grid, geometry=H)
            del H
            return grid

        grid = real_space_grid(v, grid_unit, density)
        index =  grid.index([0, 0, z])

        # Plot only the real part of the grid
        # The image will be created in an imshow layer (stored in self.imshow)
        self.imshow = self.axes.imshow(grid.grid[:, :, index[2]].T.real, cmap='seismic', origin='lower',
                              vmax=vmax, vmin=-vmax, extent=self.extent)
        # Colorbars
        if 'colorbar' in keywords:
            if keywords['colorbar'] != False:
                self.add_colorbar(self.imshow, label=label)

    def set_axes(self, bdx=2):
        g = self.HH.geom
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

    def add_colorbar(self, layer, label):
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(layer, label=label, cax=cax)
        plt.subplots_adjust(right=0.8)

    def annotate(self, size=6):
        """ Annotate the site indices in the pi-network """
        g = self.HH.geom
        x = g.xyz[:, 0]
        y = g.xyz[:, 1]
        for ia in g:
            self.axes.annotate(ia, (x[ia], y[ia]), size=size)
