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

        # Plot results
        row, col = BO.nonzero()
        for i, r in enumerate(row):
            xr = H.geom.xyz[r]
            c = col[i]
            if r < c:
                xc = H.geom.xyz[c]
                x = [xr[0], xc[0]]
                y = [xr[1], xc[1]]
                try:
                    a = int(keywords['angle'])
                except:
                    a = np.angle((xc[0]-xr[0])+1j*(xc[1]-xr[1]))*360./(2*np.pi)
                    if a > 90:
                        a -= 180
                    elif a < -90:
                        a += 180
                z = BO[r, c]
                plt.plot(x, y, c='k', ls='-', lw=4, solid_capstyle='round')
                self.axes.text(sum(x)/2, sum(y)/2, r'%.3f'%z, ha="center", va="center", rotation=a, size=6, bbox=bbox_props)
