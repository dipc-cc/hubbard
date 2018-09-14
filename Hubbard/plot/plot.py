from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Plot(object):

    def __init__(self, figsize=(8, 6)):
        self.fig = plt.figure(figsize=figsize)
        self.axes = plt.axes()
        plt.rc('font', family='Bitstream Vera Serif', size=16)
        plt.rc('text', usetex=True)

    def savefig(self, fn):
        self.fig.savefig(fn)
        print('Wrote', fn)

    def close(self):
        plt.close('all')


class GeometryPlot(Plot):

    def __init__(self, HubbardHamiltonian, **keywords):
        Plot.__init__(self)
        g = HubbardHamiltonian.geom
        x = g.xyz[:, 0]
        y = g.xyz[:, 1]
        bdx = 2
        self.xmin = min(x)-bdx
        self.xmax = max(x)+bdx
        self.ymin = min(y)-bdx
        self.ymax = max(y)+bdx
        self.extent = [min(x)-bdx, max(x)+bdx, min(y)-bdx, max(y)+bdx]
        self.axes.set_xlim(self.xmin, self.xmax)
        self.axes.set_ylim(self.ymin, self.ymax)

        # Patches
        pi = []
        aux = []
        for ia in g:
            if g.atoms[ia].Z == 1: # H
                aux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.4))
            elif g.atoms[ia].Z == 5: # B
                pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.0))
            elif g.atoms[ia].Z == 6: # C
                pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
            elif g.atoms[ia].Z == 7: # N
                pi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.0))
            elif g.atoms[ia].Z > 10: # Some other atom
                aux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.2))
        ppi = PatchCollection(pi, alpha=1., lw=1.2, edgecolor='0.6', **keywords)
        ppi.set_array(np.zeros(len(pi)))
        self.ppi = ppi
        paux = PatchCollection(aux, alpha=1., lw=1.2, edgecolor='0.6', **keywords)
        paux.set_array(np.zeros(len(aux)))
        self.paux = paux

        # Compute data
        self.axes.add_collection(self.paux)
        self.axes.add_collection(self.ppi)
        self.axes.set_xlabel(r'$x$ (\AA)')
        self.axes.set_ylabel(r'$y$ (\AA)')
        self.axes.set_aspect('equal')

    def add_colorbar(self, layer, label):
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(layer, label=label, cax=cax)
        plt.subplots_adjust(right=0.8)
