import matplotlib.pyplot as plt
from hubbard.plot import GeometryPlot
from hubbard.plot import Plot
import numpy as np
import sisl
import matplotlib as mp

__all__ = ['BondOrder', 'BondHoppings', 'Bonds']


class BondOrder(GeometryPlot):
    """ Plot the Bond order for the `hubbard.HubbardHamiltonian`

    Parameters
    ----------
    HH: hubbard.HubbardHamiltonian
        Mean-field Hubbard Hamiltonian
    """

    def __init__(self, HH, **kwargs):

        if 'cmap' not in kwargs:
            kwargs['cmap'] = plt.cm.bwr

        super().__init__(HH, **kwargs)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

        # Compute Huckel bond orders
        BO = HH.get_bond_order()
        d = 1.6 # max nearest-neighbor bond length

        # Plot results
        row, col = BO.nonzero()
        for i, r in enumerate(row):
            xr = HH.geometry.xyz[r]
            c = col[i]
            xc = HH.geometry.xyz[c]
            R = xr-xc
            if np.dot(R, R)**.5 <= d:
                # intracell bond
                x = [xr[0], xc[0]]
                y = [xr[1], xc[1]]
                self.axes.plot(x, y, c='k', ls='-', lw=4, solid_capstyle='round')
            else:
                # intercell bond
                cell = HH.geometry.sc.cell
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
                self.axes.plot(x, y, c='k', ls='-', lw=4, solid_capstyle='round')
            self.axes.text(sum(x)/2, sum(y)/2, r'%.3f'%BO[r, c], ha="center", va="center", rotation=15, size=6, bbox=bbox_props)


class BondHoppings(Plot):
    """ Plot matrix element of a tight-binding Hamiltonian in real space

    Parameters
    ----------
    H: sisl.Hamiltonian or hubbard.HubbardHamiltonian object
        tight-binding Hamitlonian
    annotate: bool, optional
        if True it annotates the numerical value of the matrix element
    off_diagonal_only: bool, optional
        if True it also plots the onsite energy
    """

    def __init__(self, H, annotate=False, off_diagonal_only=True, **kwargs):

        cmap = kwargs.get("cmap", plt.cm.jet)

        super().__init__(**kwargs)
        H = H.H.copy()
        H.set_nsc([1, 1, 1])
        tmax = np.amax(abs(H.Hk()))
        tmin = np.amin(abs(H.Hk()))
        norm = mp.colors.Normalize(vmin=tmin, vmax=tmax)
        cmap = mp.cm.ScalarMappable(norm=norm, cmap=cmap)
        for ia in H.geometry:
            x0, y0 = H.geometry.xyz[ia, 0], H.geometry.xyz[ia, 1]
            edges = H.edges(ia, exclude=ia)
            for ib in edges:
                x1, y1 = H.geometry.xyz[ib, 0], H.geometry.xyz[ib, 1]
                t = H[ia, ib, 0]
                self.axes.plot([x0, x1], [y0, y1], '-', color=cmap.to_rgba(abs(t)), linewidth=2*abs(t)/tmax, label='%.3f'%t)
                if annotate:
                    rij = H.geometry.Rij(ia, ib)
                    self.axes.annotate('%.2f'%(t), (x0+rij[0]*.5, y0+rij[1]*.5), fontsize=8)
            # Plot onsite energies?
            if not off_diagonal_only:
                self.axes.plot(x0, y0, 'or', label='%.3f'%H[ia, ia, 0])

        self.axes.axis('off')


class Bonds(Plot):
    r""" Plot bonds between atoms in geometry

    Parameters
    ----------
    G: sisl.Geometr, sisl.Hamiltonian, or hubbard.HubbardHamiltonian
        Geometry to be plotted
    annotate: bool, optional
        if True it annotates the value of the distance between atoms in the plot
    R: float or array_like
        minimum and maximum distance between atoms to consider in the plot
        Defaults to R=1.5 \AA since typically we consider sp2 carbon systems
    """

    def __init__(self, G, annotate=False, R=1.5, **kwargs):

        super().__init__(**kwargs)

        self.plot_bonds(G, annotate=annotate, R=R, **kwargs)

    def plot_bonds(self, G, annotate=False, R=0, **kwargs):
        # Define default kwarg aruments
        cmap = kwargs.get("cmap", plt.cm.jet)
        zorder = kwargs.get("zorder", 1)
        alpha = kwargs.get("alpha", 1.)
        linewidth = kwargs.get("linewidth", 2.)

        if isinstance(R, (tuple, list)):
            minR, maxR = min(R), max(R)
        else:
            minR, maxR = 0., R

        if not isinstance(G, sisl.Geometry):
            G = G.geometry
        G = G.copy()
        G.set_nsc([1, 1, 1])

        if not maxR:
            maxR = max(G.distance())
        if not minR:
            minR = min(G.distance())

        # Create colormap
        norm = mp.colors.Normalize(vmin=minR, vmax=maxR)
        cmap = mp.cm.ScalarMappable(norm=norm, cmap=cmap)
        cmap.set_array([])
        for i in G:
            x0, y0 = G.xyz[i, 0], G.xyz[i, 1]
            idx = G.close(i, R=[0.1, maxR+0.1])
            for j in idx[1]:
                x1, y1 = G.xyz[j, 0], G.xyz[j, 1]
                rij = G.Rij(i, j)
                d = np.sqrt((rij*rij).sum())
                if d <= maxR:
                    self.axes.plot([x0, x1], [y0, y1], linewidth=linewidth, color=cmap.to_rgba(d), zorder=zorder, alpha=alpha)
                    if annotate:
                        self.axes.annotate('%.2f'%(d), (x0+rij[0]*.5, y0+rij[1]*.5), fontsize=8)
         # Colorbar
        if 'colorbar' in kwargs:
            if kwargs['colorbar'] != False:
                self.cbar = self.fig.colorbar(cmap)
                if 'label' in kwargs:
                    self.cbar.ax.set_ylabel(kwargs['label'])
