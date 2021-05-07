import sisl
import matplotlib.pyplot as plt
from hubbard.plot import Plot
from hubbard.plot import GeometryPlot
import matplotlib.colors as colors
import numpy as np

__all__ = ['Spectrum', 'LDOSmap', 'DOS_distribution', 'DOS']


class Spectrum(Plot):
    """ Plot the orbital charge overlaps for the `hubbard.HubbardHamiltonian` object

    Parameters
    ----------
    HH : hubbard.HubbardHamiltonian
        the mean-field Hubbard Hamiltonian from which the LDOS should be computed
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

    def __init__(self, HH, k=[0, 0, 0], xmax=10, ymin=0, ymax=0, fontsize=16, **kwargs):

        super().__init__(**kwargs)
        self.axes.fill_between([-xmax, 0], 0, 1.0, facecolor='k', alpha=0.1)
        lmax = 0.0
        midgap = HH.find_midgap()
        for ispin in range(2):
            ev, L = HH.calc_orbital_charge_overlaps(k=k, spin=ispin)
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
    direction : 3-vector, optional
        vector defining the direction of the real-space projection
    origo : 3-vector, optional
        coordinate on the real-space projection axis
    projection : {'2D', '1D'}
        whether the projection is for the perpendicular plane (2D) or on the axis (1D)
    nx : int, optional
        number of grid points along real-space axis
    gamma_x : float, optional
        Lorentzian broadening of orbitals in real space
    dx : float, optional
        extension (in Ang) of the boundary around the system
    dist_x : {'gaussian', 'lorentzian'}
        smearing function along the real-space axis
    ne : int, optional
        number of grid points along the energy axis
    gamma_e : float, optional
        Lorentzian broadening of eigenvalues along the energy axis
    emax : float, optional
        specifies the energy range (-emax, emax) to be plotted
    dist_e : {'lorentzian', 'lorentzian'}
        smearing function along the energy axis
    vmin : float, optional
        colorscale minimum
    vmax : float, optional
        colorscale maximum
    scale : {'linear', 'log'}
        whether to use linear or logarithmic color scale
    """

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], spin=0,
                 direction=[1, 0, 0], origo=[0, 0, 0], projection='2D',
                 nx=601, gamma_x=0.5, dx=5.0, dist_x='gaussian',
                 ne=501, gamma_e=0.05, emax=10., dist_e='lorentzian',
                 vmin=0, vmax=None, scale='linear', **kwargs):

        super().__init__(**kwargs)
        ev, evec = HubbardHamiltonian.eigh(k=k, eigvals_only=False, spin=spin)
        ev -= HubbardHamiltonian.find_midgap()
        xyz = np.array(HubbardHamiltonian.geometry.xyz[:])
        # coordinates relative to selected origo
        xyz -= np.array(origo).reshape(1, 3)
        # distance along projection axis
        unitvec = np.array(direction)
        unitvec = unitvec / unitvec.dot(unitvec) ** 0.5
        coord = xyz.dot(unitvec)

        xmin, xmax = min(coord) - dx, max(coord) + dx
        emin, emax = -emax, emax

        # Broaden along real-space axis
        x = np.linspace(xmin, xmax, nx)
        dist_x = sisl.get_distribution(dist_x, smearing=gamma_x)
        xcoord = x.reshape(-1, 1) - coord.reshape(1, -1) # (nx, natoms)
        if projection.upper() == '1D':
            # distance perpendicular to projection axis
            perp = xyz - coord.reshape(-1, 1) * unitvec
            perp = np.einsum('ij,ij->i', perp, perp)
            xcoord = (xcoord ** 2 + perp) ** 0.5
        DX = dist_x(xcoord)

        # Broaden along energy axis
        e = np.linspace(emin, emax, ne)
        dist_e = sisl.get_distribution(dist_e, smearing=gamma_e)
        DE = dist_e(e.reshape(-1, 1) - ev.reshape(1, -1)) # (ne, norbs)

        # Compute DOS
        prob_dens = np.abs(evec) ** 2
        DOS = DX.dot(prob_dens).dot(DE.T)
        intdat = np.sum(DOS) * (x[1] - x[0]) * (e[1] - e[0])
        print('Integrated LDOS spectrum (states within plot):', intdat, DOS.shape)

        cm = plt.cm.hot
        if scale == 'log':
            if vmin == 0:
                vmin = 1e-4
            norm = colors.LogNorm(vmin=vmin)
        else:
            # Linear scale
            norm = colors.Normalize(vmin=vmin)
        self.imshow = self.axes.imshow(DOS.T, extent=[xmin, xmax, emin, emax], cmap=cm, \
                                       origin='lower', norm=norm, vmax=vmax)
        title = f'LDOS projection in {projection.upper()}'
        if projection.upper() == '1D':
            title += r': origo [%.2f,%.2f,%.2f] (\AA)' % tuple(origo)
        self.set_title(title)
        self.set_xlabel(r'distance along [%.2f,%.2f,%.2f] (\AA)' % tuple(direction))
        self.set_ylabel(r'$E-E_\mathrm{midgap}$ (eV)')
        self.set_xlim(xmin, xmax)
        self.set_ylim(emin, emax)
        self.axes.set_aspect('auto')


class DOS_distribution(GeometryPlot):
    """ Plot LDOS in the configuration space for the `hubbard.HubbardHamiltonian` object

    Parameters
    ----------
    HH: HubbardHamiltonian
        mean-field Hubbard Hamiltonian
    PDOS: numpy.ndarray
        Density of states projected in atomic sites obtained at certain energy
    realspace:
        If True it will plot the DOS in a realspace grid otherwise it plots it as a scatter plot
        with varying size depending on the PDOS numerical value

    See Also
    ------------
    hubbard.HubbardHamiltonian.PDOS
    """

    def __init__(self, HH, PDOS, sites=[], ext_geom=None, realspace=False, **kwargs):

        # Set default kwargs
        if realspace:
            if 'facecolor' not in kwargs:
                kwargs['facecolor'] = 'None'
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'Greys'
        else:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.bwr

        super().__init__(HH.geometry, ext_geom=ext_geom, **kwargs)

        x = HH.geometry[:, 0]
        y = HH.geometry[:, 1]

        if realspace:
            if 'vmin' not in kwargs:
                kwargs['vmin'] = 0
            self.__realspace__(PDOS, density=True, **kwargs)
            self.imshow.set_cmap(plt.cm.afmhot)

        else:
            self.axes.scatter(x, y, PDOS, 'b')

        for i, s in enumerate(sites):
            self.axes.text(x[s], y[s], '%i' % i, fontsize=15, color='r')


class DOS(Plot):
    """ Plot the total density of states (DOS) as a function of the energy for the `hubbard.HubbardHamiltonian` object

    Parameters
    ----------
    HH: HubbardHamiltonian
        mean-field Hubbard Hamiltonian
    egrid: array_like
        energy grid to compute the density of states
    eta: float, optional
        smearing parameter to compute DOS
    spin: int or array_like, optional
        plot DOS for selected spin index or sum them both if ``spin=[0,1]``
    sites: array_like, optional
        sum projected DOS into selected atomic sites

    See Also
    ------------
    hubbard.HubbardHamiltonian.DOS
    hubbard.HubbardHamiltonian.PDOS
    """

    def __init__(self, HH, egrid, eta=1e-3, spin=[0, 1], sites=[], **kwargs):

        super().__init__(**kwargs)

        if np.any(sites):
            DOS = HH.PDOS(egrid, eta=eta, spin=spin)
            offset = 0. * np.average(DOS[sites[0]])
            for i, s in enumerate(sites):
                self.axes.plot(egrid, DOS[s] + offset * i, label='site %i' % i)
            self.legend()
        else:
            DOS = HH.DOS(egrid, eta=eta, spin=spin)
            self.axes.plot(egrid, DOS, label='TDOS')

        self.set_xlabel(r'E-E$_\mathrm{midgap}$ [eV]')
        self.set_ylabel(r'DOS [1/eV]')
