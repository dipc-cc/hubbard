import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
from Hubbard.plot import Plot
import numpy as np
import matplotlib as mp

__all__ = ['BondOrder', 'BondHoppings', 'Bonds']


class BondOrder(GeometryPlot):
    """ Plot the Bond order for the HubbardHamiltonian """

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
            xr = H.geometry.xyz[r]
            c = col[i]
            xc = H.geometry.xyz[c]
            R = xr-xc
            if np.dot(R, R)**.5 <= d:
                # intracell bond
                x = [xr[0], xc[0]]
                y = [xr[1], xc[1]]
                plt.plot(x, y, c='k', ls='-', lw=4, solid_capstyle='round')
            else:
                # intercell bond
                cell = H.geometry.sc.cell
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


class BondHoppings(Plot):
    """ Plot matrix element of sisl.Hamiltonian. Only off-diagonal elements """

    def __init__(self, H, annotate=False, **keywords):

        if 'cmap' not in keywords:
            cm = plt.cm.jet
        else:
            cm = keywords['cmap']

        Plot.__init__(self, **keywords)
        H = H.H
        H.set_nsc([1, 1, 1])
        tmax = np.amax(abs(H.Hk()))
        tmin = np.amin(abs(H.Hk()))
        norm = mp.colors.Normalize(vmin=tmin, vmax=tmax)
        cmap = mp.cm.ScalarMappable(norm=norm, cmap=cm)
        for ia in H.geometry:
            x0, y0 = H.geometry.xyz[ia, 0], H.geometry.xyz[ia, 1]
            edges = H.edges(ia)
            for ib in edges[edges!=ia]:
                x1, y1 = H.geometry.xyz[ib, 0], H.geometry.xyz[ib, 1]
                t = H[ia, ib, 0]
                self.axes.plot([x0, x1], [y0, y1], '-', color=cmap.to_rgba(abs(t)), linewidth=2*abs(t)/tmax, label='%.3f'%t)
                if annotate:
                    rij = H.geometry.Rij(ia, ib)
                    self.axes.annotate('%.2f'%(t), (x0+rij[0]*.5, y0+rij[1]*.5), fontsize=8)

        self.axes.axis('off')


class Bonds(Plot):
    """ Plot bonds between atoms in geometry """

    def __init__(self, H0, annotate=False, maxR=0, minR=0, **keywords):

        if 'cmap' not in keywords:
            cm = plt.cm.jet
        else:
            cm = keywords['cmap']

        Plot.__init__(self, **keywords)
        H = H0.copy()
        H.H.set_nsc([1, 1, 1])

        if not maxR:
            maxR = max(H.geometry.distance())
        if not minR:
            minR = min(H.geometry.distance())

        # Create colormap
        norm = mp.colors.Normalize(vmin=minR, vmax=maxR)
        cmap = mp.cm.ScalarMappable(norm=norm, cmap=cm)
        cmap.set_array([])
        for i in H.geometry:
            x0, y0 = H.geometry.xyz[i, 0], H.geometry.xyz[i, 1]
            edges = H.H.edges(i)
            for j in edges:
                x1, y1 = H.geometry.xyz[j, 0], H.geometry.xyz[j, 1]
                rij = H.geometry.Rij(i, j)
                d = np.sqrt((rij*rij).sum())
                self.axes.plot([x0, x1], [y0, y1], linewidth=2, color=cmap.to_rgba(d))
                if annotate:
                    self.axes.annotate('%.2f'%(d), (x0+rij[0]*.5, y0+rij[1]*.5), fontsize=8)
         # Colorbar
        if 'colorbar' in keywords:
            if keywords['colorbar'] != False:
                self.cbar = self.fig.colorbar(cmap)
                if 'label' in keywords:
                    self.cbar.ax.set_ylabel(keywords['label'])
