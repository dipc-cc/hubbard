from __future__ import print_function

import matplotlib.pyplot as plt
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

    def __init__(self, HubbardHamiltonian, figsize=(8, 6)):
        Plot.__init__(self, figsize=figsize)
        x = HubbardHamiltonian.geom.xyz[:, 0]
        y = HubbardHamiltonian.geom.xyz[:, 1]
        bdx = 2
        self.axes.set_xlim(min(x)-bdx, max(x)+bdx)
        self.axes.set_ylim(min(y)-bdx, max(y)+bdx)
        # Patches
        self.ppi, self.paux = HubbardHamiltonian.get_atomic_patches()
        # Compute data
        self.axes.add_collection(self.paux)
        self.axes.add_collection(self.ppi)
        self.axes.set_xlabel(r'$x$ (\AA)')
        self.axes.set_ylabel(r'$y$ (\AA)')
        self.axes.set_aspect('equal')

    def add_colorbar(self, label):
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(self.ppi, label=label, cax=cax)
        plt.subplots_adjust(right=0.8)
