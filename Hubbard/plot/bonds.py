from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
from Hubbard.plot import Plot
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
                # Compute projections onto lattice vectors
                P = np.dot(cell, R)
                # Normalize
                for i in range(3):
                    P[i] /= np.dot(cell[i], cell[i])**.5
                # Look up largest projection
                j = np.argmax(np.abs(P))
                R -= np.sign(P[j])*cell[j] # Subtract relevant lattice vector
                x = [xr[0], xr[0]-R[0]/2]
                y = [xr[1], xr[1]-R[1]/2]
                plt.plot(x, y, c='k', ls='-', lw=4, solid_capstyle='round')
            self.axes.text(sum(x)/2, sum(y)/2, r'%.3f'%BO[r, c], ha="center", va="center", rotation=15, size=6, bbox=bbox_props)

class bonds_hoppings(Plot):

    def __init__(self, H, **keywords):
        
        Plot.__init__(self, **keywords)

        for ia in H.geom:
            x0, y0 = H.geom.xyz[ia, 0], H.geom.xyz[ia, 1] 
            for ib in H.geom:
                x1, y1 = H.geom.xyz[ib, 0], H.geom.xyz[ib, 1] 
                t = H[ia, ib][0,0]
                if abs(t) == 2.7:
                    plt.plot([x0,x1], [y0,y1], '-', markersize=2, color='blue', linewidth=1.2)
                elif abs(t) == 0.2:
                    plt.plot([x0,x1], [y0,y1], '--', markersize=2, color='red', linewidth=1.2)
                elif abs(t) == 0.18:
                    plt.plot([x0,x1], [y0,y1], '--', markersize=2, color='green', linewidth=1.2)
        
        plt.axis('off')
