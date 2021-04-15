import matplotlib.pyplot as plt
from hubbard.plot import Plot
from hubbard.plot import GeometryPlot
import matplotlib.colors as colors
import numpy as np

__all__ = ['Spectrum', 'LDOSmap', 'DOS_distribution', 'DOS']


class Spectrum(Plot):
    """ Plot the orbital charge overlaps for the `HubbardHamiltonian` object

    Parameters
    ----------
    HubbardHamiltonian : HubbardHamiltonian
        the HubbardHamiltonian from which the LDOS should be computed
    k : array_like, optional
        k-point in the Brillouin zone to sample
    xmax : float, optional
        the energy range (-xmax, xmax) wrt. midgap to be plotted
    ymin : float, optional
        the y-axis minimum
    ymax : float, optional
        the y-axis maximum
    fontsize : float, optional
        fontsize
    """

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], xmax=10, ymin=0, ymax=0, fontsize=16, **kwargs):

        super().__init__(**kwargs)
        self.axes.fill_between([-xmax, 0], 0, 1.0, facecolor='k', alpha=0.1)
        lmax = 0.0
        midgap = HubbardHamiltonian.find_midgap()
        for ispin in range(2):
            ev, L = HubbardHamiltonian.calc_orbital_charge_overlaps(k=k, spin=ispin)
            ev -= midgap
            L = np.diagonal(L)
            lmax = max(max(L), lmax)
            self.axes.plot(ev, L, 'rg'[ispin] + '.+'[ispin], label=[r'$\sigma=\uparrow$', r'$\sigma=\downarrow$'][ispin])

            if 'annotate' in kwargs:
                if kwargs['annotate'] != False:
                    for i in range(len(ev)):
                        if np.abs(ev[i]) < xmax:
                            self.axes.annotate(f'({i}, {ev[i]:.3f})', (ev[i], L[i] * 1.05), fontsize=6, rotation=45)
        self.axes.legend()
        self.set_xlabel(r'$E_{\alpha\sigma}-E_\mathrm{mid}$ (eV)', fontsize=fontsize)
        self.set_ylabel(r'$\eta_{\alpha\sigma}=\int dr |\psi_{\alpha\sigma}|^4$', fontsize=fontsize)
        self.set_xlim(-xmax, xmax)
        if ymax == 0:
            self.set_ylim(ymin, lmax + 0.01)
        else:
            self.set_ylim(ymin, ymax)


class LDOSmap(Plot):
    """ Plot LDOS(distance, energy) map resolved in energy and axis-coordinates for the `HubbardHamiltonian` object

    Parameters
    ----------

    HubbardHamiltonian : HubbardHamiltonian
        the HubbardHamiltonian from which the LDOS should be computed
    k : array_like, optional
        k-point in the Brillouin zone to sample
    spin : int, optional
        spin index
    axis : int, optional
        real-space index along which LDOS is resolved
    nx : int, optional
        number of grid points along real-space axis
    gamma_x : float, optional
        Lorentzian broadening of orbitals along the real-space axis
    dx : float, optional
        extension (in Ang) of the boundary around the system
    ny : int, optiona
        number of grid points along the energy axis
    gamma_e : float, optional
        Lorentzian broadening of eigenvalues along the energy axis
    ymax : float, optional
        specifies the energy range (-ymax, ymax) to be plotted
    vmin : float, optional
        colorscale minimum
    vmax : float, optional
        colorscale maximum
    scale : {'linear', 'log'}
        whether to use linear or logarithmic color scale
    """

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], spin=0, axis=0,
                 nx=501, gamma_x=1.0, dx=5.0, ny=501, gamma_e=0.05, ymax=10., vmin=0, vmax=None, scale='linear',
                 **kwargs):

        super().__init__(**kwargs)
        ev, evec = HubbardHamiltonian.eigh(k=k, eigvals_only=False, spin=spin)
        ev -= HubbardHamiltonian.find_midgap()
        coord = HubbardHamiltonian.geometry.xyz[:, axis]

        xmin, xmax = min(coord) - dx, max(coord) + dx
        ymin, ymax = -ymax, ymax
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        dat = np.zeros((len(x), len(y)))
        for i, evi in enumerate(ev):
            de = gamma_e / ((y - evi) ** 2 + gamma_e ** 2) / np.pi
            dos = np.zeros(len(x))
            for j, vj in enumerate(evec[:, i]):
                dos += abs(vj) ** 2 * gamma_x / ((x - coord[j]) ** 2 + gamma_x ** 2) / np.pi
            dat += np.outer(dos, de)
        intdat = np.sum(dat) * (x[1] - x[0]) * (y[1] - y[0])
        print('Integrated LDOS spectrum (states within plot):', intdat)
        cm = plt.cm.hot

        if scale == 'log':
            if vmin == 0:
                vmin = 1e-4
            norm = colors.LogNorm(vmin=vmin)
        else:
            # Linear scale
            norm = colors.Normalize(vmin=vmin)
        self.imshow = self.axes.imshow(dat.T, extent=[xmin, xmax, ymin, ymax], cmap=cm, \
                                       origin='lower', norm=norm, vmax=vmax)
        if axis == 0:
            self.set_xlabel(r'$x$ (\AA)')
        elif axis == 1:
            self.set_xlabel(r'$y$ (\AA)')
        elif axis == 2:
            self.set_xlabel(r'$z$ (\AA)')
        self.set_ylabel(r'$E-E_\mathrm{midgap}$ (eV)')
        self.set_xlim(xmin, xmax)
        self.set_ylim(ymin, ymax)
        self.axes.set_aspect('auto')


class DOS_distribution(GeometryPlot):
    """ Plot LDOS in the configuration space for the `HubbardHamiltonian` object

    Notes
    -----
    If the `realspace` kwarg is passed it will plot the DOS in a realspace grid
    """

    def __init__(self, HubbardHamiltonian, DOS, sites=[], ext_geom=None, realspace=False, **kwargs):

        # Set default kwargs
        if realspace:
            if 'facecolor' not in kwargs:
                kwargs['facecolor'] = 'None'
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'Greys'
        else:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.bwr

        super().__init__(HubbardHamiltonian.geometry, ext_geom=ext_geom, **kwargs)

        x = HubbardHamiltonian.geometry[:, 0]
        y = HubbardHamiltonian.geometry[:, 1]

        if realspace:
            if 'vmin' not in kwargs:
                kwargs['vmin'] = 0
            self.__realspace__(DOS, density=True, **kwargs)
            self.imshow.set_cmap(plt.cm.afmhot)

        else:
            self.axes.scatter(x, y, DOS, 'b')

        for i, s in enumerate(sites):
            self.axes.text(x[s], y[s], '%i' % i, fontsize=15, color='r')


class DOS(Plot):
    """ Plot the total DOS as a function of the energy for the `HubbardHamiltonian` object """

    def __init__(self, HubbardHamiltonian, egrid, eta=1e-3, spin=[0, 1], sites=[], **kwargs):

        super().__init__(**kwargs)

        if np.any(sites):
            DOS = HubbardHamiltonian.PDOS(egrid, eta=eta, spin=spin)
            offset = 0. * np.average(DOS[sites[0]])
            for i, s in enumerate(sites):
                self.axes.plot(egrid, DOS[s] + offset * i, label='site %i' % i)
            self.legend()
        else:
            DOS = HubbardHamiltonian.DOS(egrid, eta=eta, spin=spin)
            self.axes.plot(egrid, DOS, label='TDOS')

        self.set_xlabel(r'E-E$_\mathrm{midgap}$ [eV]')
        self.set_ylabel(r'DOS [1/eV]')
