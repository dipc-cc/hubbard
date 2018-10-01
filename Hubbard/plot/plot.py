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
        if 'title' in keywords:
            self.set_title(keywords['title'])
        plt.rc('font', family='Bitstream Vera Serif', size=16)
        plt.rc('text', usetex=True)

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

class GeometryPlot(Plot):

    def __init__(self, HubbardHamiltonian, **keywords):
        Plot.__init__(self, **keywords)
        self.geom = HubbardHamiltonian.geom
        self.pi_geom = HubbardHamiltonian.pi_geom
        g = self.geom
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

        # Relevant keywords
        kw = {}
        for k in keywords:
            if k in ['cmap']:
                kw[k] = keywords[k]

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
        # Pi sites
        ppi = PatchCollection(pi, alpha=1., lw=1.2, edgecolor='0.6', **kw)
        ppi.set_array(np.zeros(len(pi)))
        ppi.set_clim(-1, 1)
        self.ppi = ppi
        # Aux sites
        paux = PatchCollection(aux, alpha=1., lw=1.2, edgecolor='0.6', **kw)
        paux.set_array(np.zeros(len(aux)))
        paux.set_clim(-1, 1)
        self.paux = paux

        self.axes.add_collection(self.paux)
        self.axes.add_collection(self.ppi)
        self.set_xlabel(r'$x$ (\AA)')
        self.set_ylabel(r'$y$ (\AA)')
        self.axes.set_aspect('equal')

    def add_colorbar(self, layer, label):
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(layer, label=label, cax=cax)
        plt.subplots_adjust(right=0.8)

    def annotate(self, size=6):
        g = self.pi_geom
        x = g.xyz[:, 0]
        y = g.xyz[:, 1]
        for ia in g:
            self.axes.annotate(ia, (x[ia], y[ia]), size=size)
