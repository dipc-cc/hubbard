import sisl
import matplotlib.pyplot as plt
from hubbard.plot import Plot
from hubbard.plot import GeometryPlot
import matplotlib.colors as colors
import numpy as np
from hubbard.grid import *

__all__ = ['Spectrum', 'DOSmap', 'LDOS_from_eigenstate', 'LDOS', 'PDOS']


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
        for ispin in range(HH.spin_size):
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


class DOSmap(Plot):
    """ Plot DOS(distance, energy) map resolved in energy and axis-coordinates for the `HubbardHamiltonian` object

    Parameters
    ----------

    HubbardHamiltonian : HubbardHamiltonian
        the HubbardHamiltonian from which the DOS should be computed
    k : array_like, optional
        k-point in the Brillouin zone to sample
    spin : int, optional
        spin index
    direction : 3-vector, optional
        vector defining the direction of the real-space projection
    origin : 3-vector, optional
        coordinate on the real-space projection axis
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
                 direction=[1, 0, 0], origin=[0, 0, 0],
                 nx=601, gamma_x=0.5, dx=5.0, dist_x='gaussian',
                 ne=501, gamma_e=0.05, emax=10., dist_e='lorentzian',
                 vmin=0, vmax=None, scale='linear', **kwargs):

        super().__init__(**kwargs)
        ev, evec = HubbardHamiltonian.eigh(k=k, eigvals_only=False, spin=spin)
        ev -= HubbardHamiltonian.find_midgap()
        xyz = np.array(HubbardHamiltonian.geometry.xyz[:])
        # coordinates relative to selected origin
        xyz -= np.array(origin).reshape(1, 3)
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

        DX = dist_x(xcoord)

        # Broaden along energy axis
        e = np.linspace(emin, emax, ne)
        dist_e = sisl.get_distribution(dist_e, smearing=gamma_e)
        DE = dist_e(e.reshape(-1, 1) - ev.reshape(1, -1)) # (ne, norbs)

        # Compute DOS
        prob_dens = np.abs(evec) ** 2
        DOS = DX.dot(prob_dens).dot(DE.T)
        intdat = np.sum(DOS) * (x[1] - x[0]) * (e[1] - e[0])
        print('Integrated DOS spectrum (states within plot):', intdat, DOS.shape)

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

        self.set_xlabel(r'distance along [%.2f,%.2f,%.2f] (\AA)' % tuple(direction))
        self.set_ylabel(r'$E-E_\mathrm{midgap}$ (eV)')
        self.set_xlim(xmin, xmax)
        self.set_ylim(emin, emax)
        self.axes.set_aspect('auto')


class LDOS_from_eigenstate(GeometryPlot):
    """ Plot LDOS in the configuration space for the `hubbard.HubbardHamiltonian` object

    Parameters
    ----------
    HH: HubbardHamiltonian
        mean-field Hubbard Hamiltonian
    WF: numpy.ndarray or sisl.physics.electron.EigenstateElectron
        Eigenstate to be plotted as LDOS, WF can be an array-like containing more than WF
        The WF have to be stored along the second axis, i.e. each WF is stored as column vectors
        in this case this method sums the EigenLDOS corresponding to each WF in the grid
        This can be handy if one wants to plot the LDOS = LDOS_up + LDOS_dn
    realspace:
        If True it will plot the LDOS in a realspace grid otherwise it plots it as a scatter plot (PDOS)
        with varying size depending on the PDOS numerical value
        In this case either a `sisl.SuperCell` (`sc` kwarg) or the `z` kwarg to slice the real space grid at the desired z coordinate needs to be passed
    """

    def __init__(self, HH, wavefunction, sites=[], ext_geom=None, realspace=False, **kwargs):

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

        # Ensure WF is an array of correct dimensions
        WF = np.array(wavefunction)
        if WF.shape == (len(WF),):
            WF = np.expand_dims(WF, axis=1)
        if realspace:
            if 'shape' not in kwargs:
                kwargs['shape'] = [100,100,1]

            if 'vmin' not in kwargs:
                kwargs['vmin'] = 0

            xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

            if 'sc' not in kwargs:
                if 'z' in kwargs:
                    origin = [xmin, ymin, -kwargs['z']]
                else:
                    raise ValueError('Either a SC or the z coordinate to slice the real space grid needs to be passed')

                kwargs['sc'] = sisl.SuperCell([xmax-xmin, ymax-ymin, 1000], origin=origin)


            if 'axis' not in kwargs:
                kwargs['axis'] = 2

            grid = 0
            for i in range(WF.shape[1]):
                dos = WF[:,i]
                grid_i = real_space_grid(self.geometry, kwargs['sc'], dos, kwargs['shape'], mode='wavefunction')
                # Slice it to obtain a 2D grid
                slice_grid = grid_i.swapaxes(kwargs['axis'], 2).grid[:,:,0].T
                grid += slice_grid.real**2 + slice_grid.imag**2

            self.__realspace__(grid, **kwargs)
            self.imshow.set_cmap(plt.cm.afmhot)

        else:
            dos = 0
            for i in range(WF.shape[1]):
                wf = WF[:,i]
                dos += wf ** 2

            self.axes.scatter(x, y, dos, 'b')

        for i, s in enumerate(sites):
            self.axes.text(x[s], y[s], '%i' % i, fontsize=15, color='r')

class LDOS(GeometryPlot):
    """ Plot LDOS for an energy E with a certain distribution function for the `hubbard.HubbardHamiltonian` object

    Parameters
    ----------
    HH: HubbardHamiltonian
        mean-field Hubbard Hamiltonian
    E: float
        energy at which the LDOS will be computed
    eta: float, optional
        smearing parameter. Defaults to 1e-3 eV
    dist: str, optional
        distribution function. Defaults to Lorentzian distribution
    spin: array_like, optional
        To plot the LDOS corresponding to the specified spin index. Default to ``spin=[0,1]``
        i.e. it sums the LDOS corresponding to both spin components
    realspace:
        If True it will plot the LDOS in a realspace grid otherwise it plots it as a scatter plot (PDOS)
        with varying size depending on the PDOS numerical value
        In this case either a `sisl.SuperCell` (`sc` kwarg) or the `z` kwarg to slice the real space grid at the desired z coordinate needs to be passed

    See Also
    ------------
    hubbard.HubbardHamiltonian.PDOS
    sisl.get_distribution: sisl method to create distribution function
    sisl.physics.electron.PDOS: sisl method to obtain PDOS
    """

    def __init__(self, HH, E, sites=[], spin=[0,1], ext_geom=None, realspace=False, eta=1e-3, distribution='lorentzian', **kwargs):

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
            if 'shape' not in kwargs:
                kwargs['shape'] = [100,100,1]

            if 'vmin' not in kwargs:
                kwargs['vmin'] = 0

            xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

            if 'sc' not in kwargs:
                if 'z' in kwargs:
                    origin = [xmin, ymin, -kwargs['z']]
                else:
                    raise ValueError('Either a SC or the z coordinate to slice the real space grid needs to be passed')

                kwargs['sc'] = sisl.SuperCell([xmax-xmin, ymax-ymin, 1000], origin=origin)

            if 'axis' not in kwargs:
                kwargs['axis'] = 2

            grid = 0
            for s in spin:
                ev, evec = HH.eigh(spin=s, eigvals_only=False)

                if 'energy_window' in kwargs:
                    energy_window = kwargs['energy_window']
                    window_states = np.where(np.abs(ev-E)<energy_window)[0]
                else:
                    window_states = range(HH.sites)

                # Computing grids for all states is too slow... Let's use a window
                energy_window = np.where(np.abs(ev-E)<1.5)[0]
                for ni in window_states:
                    v = evec[:,ni]
                    grid_n = real_space_grid(self.geometry, kwargs['sc'], v, kwargs['shape'], mode='wavefunction')
                    f = sisl.get_distribution(distribution, smearing=eta, x0=ev[ni])
                    weight = f(E)
                    # Slice it to obtain a 2D grid
                    slice_grid = grid_n.swapaxes(kwargs['axis'], 2).grid[:, :, 0].T
                    grid += (slice_grid.real**2 + slice_grid.imag**2) * weight

            self.__realspace__(grid, **kwargs)
            self.imshow.set_cmap(plt.cm.afmhot)

        else:
            pdos = HH.PDOS(E, eta=eta, spin=spin, dist=distribution)
            self.axes.scatter(x, y, pdos, 'b')

        for i, s in enumerate(sites):
            self.axes.text(x[s], y[s], '%i' % i, fontsize=15, color='r')

class PDOS(Plot):
    """ Plot the projected density of states (PDOS) onto a subset of sites as a function of the energy for the `hubbard.HubbardHamiltonian` object

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
        by default it projects onto all sites

    See Also
    ------------
    hubbard.HubbardHamiltonian.DOS
    hubbard.HubbardHamiltonian.PDOS
    sisl.get_distribution: sisl method to create distribution function
    sisl.physics.electron.PDOS: sisl method to obtain PDOS
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
