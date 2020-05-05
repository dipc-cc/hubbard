import matplotlib.pyplot as plt
from Hubbard.plot import Plot
from Hubbard.plot import GeometryPlot
import matplotlib.colors as colors
import numpy as np

__all__ = ['Spectrum', 'LDOSmap', 'DOS_distribution', 'DOS']


class Spectrum(Plot):
    """ Plot the orbital charge overlaps for the `HubbardHamiltonian` object """

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], xmax=10, ymin=0, ymax=0, fontsize=16, **keywords):

        Plot.__init__(self, **keywords)
        self.axes.fill_between([-xmax, 0], 0, 1.0, facecolor='k', alpha=0.1)
        lmax = 0.0
        HubbardHamiltonian.find_midgap()
        for ispin in range(2):
            ev, L = HubbardHamiltonian.calc_orbital_charge_overlaps(k=k, spin=ispin)
            ev -= HubbardHamiltonian.midgap
            L = np.diagonal(L)
            lmax = max(max(L), lmax)
            plt.plot(ev, L, 'rg'[ispin]+'.+'[ispin], label=[r'$\sigma=\uparrow$', r'$\sigma=\downarrow$'][ispin])

            if 'annotate' in keywords:
                if keywords['annotate'] != False:
                    for i in range(len(ev)):
                        self.axes.annotate(i, (ev[i], L[i]), fontsize=6)
        self.axes.legend()
        self.set_xlabel(r'$E_{\alpha\sigma}-E_\mathrm{mid}$ (eV)', fontsize=fontsize)
        self.set_ylabel(r'$\eta_{\alpha\sigma}=\int dr |\psi_{\alpha\sigma}|^4$', fontsize=fontsize)
        self.set_xlim(-xmax, xmax)
        if ymax == 0:
            self.set_ylim(ymin, lmax+0.01)
        else:
            self.set_ylim(ymin, ymax)


class LDOSmap(Plot):
    """ Plot LDOS map resolved in energy and axis-coordinates for the `HubbardHamiltonian` object """

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], spin=0, axis=0,
                 nx=501, gamma_x=1.0, dx=5.0, ny=501, gamma_e=0.05, ymax=10., vmin=0, vmax=None, scale='linear',
                 **keywords):

        Plot.__init__(self, **keywords)
        ev, evec = HubbardHamiltonian.eigh(k=k, eigvals_only=False, spin=spin)
        ev -= HubbardHamiltonian.midgap
        coord = HubbardHamiltonian.geom.xyz[:, axis]

        xmin, xmax = min(coord)-dx, max(coord)+dx
        ymin, ymax = -ymax, ymax
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        dat = np.zeros((len(x), len(y)))
        for i, evi in enumerate(ev):
            de = gamma_e/((y-evi)**2+gamma_e**2)/np.pi
            dos = np.zeros(len(x))
            for j, vj in enumerate(evec[:, i]):
                dos += abs(vj)**2*gamma_x/((x-coord[j])**2+gamma_x**2)/np.pi
            dat += np.outer(dos, de)
        intdat = np.sum(dat)*(x[1]-x[0])*(y[1]-y[0])
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
        if axis==0:
            self.set_xlabel(r'$x$ (\AA)')
        elif axis==1:
            self.set_xlabel(r'$y$ (\AA)')
        elif axis==2:
            self.set_xlabel(r'$z$ (\AA)')
        self.set_ylabel(r'$E-E_\mathrm{midgap}$ (eV)')
        self.set_xlim(xmin, xmax)
        self.set_ylim(ymin, ymax)
        self.axes.set_aspect('auto')


class DOS_distribution(GeometryPlot):
    """ Plot LDOS in the configuration space for the `HubbardHamiltonian` object

    Notes
    -----
    If the `realspace` keyword is passed it will plot the DOS in a realspace grid
    """

    def __init__(self, HubbardHamiltonian, DOS, sites=[], ext_geom=None, realspace=False, **keywords):

        # Set default keywords
        if realspace:
            if 'facecolor' not in keywords:
                keywords['facecolor'] = 'None'
            if 'cmap' not in keywords:
                keywords['cmap'] = 'Greys'
        else:
            if 'cmap' not in keywords:
                keywords['cmap'] = plt.cm.bwr

        GeometryPlot.__init__(self, HubbardHamiltonian.geom, ext_geom=ext_geom, **keywords)

        x = HubbardHamiltonian.geom[:, 0]
        y = HubbardHamiltonian.geom[:, 1]

        if realspace:
            self.__realspace__(DOS, density=True, vmin=0, **keywords)
            self.imshow.set_cmap(plt.cm.afmhot)

        else:
            self.axes.scatter(x, y, DOS, 'b')

        for i, s in enumerate(sites):
            self.axes.text(x[s], y[s], '%i'%i, fontsize=15, color='r')


class DOS(Plot):
    """ Plot the total DOS as a function of the energy for the `HubbardHamiltonian` object """

    def __init__(self, HubbardHamiltonian, egrid, eta=1e-3, spin=[0, 1], sites=[], **keywords):

        Plot.__init__(self, **keywords)

        DOS = HubbardHamiltonian.DOS(egrid, eta=eta, spin=spin)

        if np.any(sites):
            offset = 0.*np.average(DOS[sites[0]])
            for i, s in enumerate(sites):
                self.axes.plot(egrid, DOS[s]+offset*i, label='%i'%i)

        else:
            TDOS = DOS.sum(axis=0)
            plt.plot(egrid, TDOS, label='TDOS')

        self.set_xlabel(r'E-E$_\mathrm{midgap}$ [eV]')
        self.set_ylabel(r'DOS [1/eV]')
        self.axes.legend()
