from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
import numpy as np


class BondOrder(GeometryPlot):

    def __init__(self, HubbardHamiltonian, **keywords):
        
        H = HubbardHamiltonian
        if 'cmap' not in keywords:
            keywords['cmap'] = plt.cm.bwr

        GeometryPlot.__init__(self, H, **keywords)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

        # Compute Huckel bond orders
        BO = H.get_bond_order()
        d = 1.6 # max nearest-neighbor bond length

        # Plot results
        row, col = BO.nonzero()
        for i, r in enumerate(row):
            xr = H.geom.xyz[r]
            c = col[i]
            xc = H.geom.xyz[c]
            R = xr-xc
            if np.dot(R, R)**.5 <= d:
                # intracell bond
                x = [xr[0], xc[0]]
                y = [xr[1], xc[1]]
                plt.plot(x, y, c='k', ls='-', lw=4, solid_capstyle='round')
            else:
                # intercell bond
                cell = H.geom.sc.cell
                P = np.dot(cell, R)
                j = np.argmax(np.abs(P))
                R -= np.sign(P[j])*cell[j] # Subtract relevant lattice vector
                x = [xr[0], xr[0]-R[0]/2]
                y = [xr[1], xr[1]-R[1]/2]
                plt.plot(x, y, c='k', ls='-', lw=4, solid_capstyle='round')
            self.axes.text(sum(x)/2, sum(y)/2, r'%.3f'%BO[r, c], ha="center", va="center", rotation=15, size=6, bbox=bbox_props)
