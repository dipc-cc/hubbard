import matplotlib.pyplot as plt
from hubbard.plot import GeometryPlot
import sisl
import numpy as np
from hubbard.grid import *

__all__ = ['Charge', 'ChargeDifference', 'SpinPolarization']


class Charge(GeometryPlot):
    r""" Plot the total charge for the `hubbard.HubbardHamiltonian` object

    :math:`\langle n_{\uparrow}\rangle+\langle n_{\downarrow}\rangle`

    or the charge for a particular spin-channel (:math:`\langle n_{\uparrow}\rangle`)
    by specifying the `spin` index.

    Parameters
    ----------
    HH: hubbard.HubbardHamiltonian
        mean-field Hubbard Hamiltonian
    ext_geom: sisl.Geometry, optional
        usually sp2 carbon-like systems are saturated with Hydrogen atoms, that don't play
        a role in the Hubbard Hamiltonian. If ext_geom is passed it plots the full sp2 carbon system
        otherwise it only uses the geometry associated to the `hubbard.HubbardHamiltonian` (carbon backbone)
    spin: int or array_like, optional:
        plot charge associated to one specific spin index, or both if ``spin=[0,1]``
    realspace: bool, optional
        if True it plots the total charge in a realspace grid. In other case it will be plotted as Mulliken populations

    """

    def __init__(self, HH, ext_geom=None, spin=[0, 1], realspace=False, **kwargs):
        # Set default kwargs
        if realspace:
            if 'facecolor' not in kwargs:
                kwargs['facecolor'] = 'None'
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'Greys'
            if 'label' not in kwargs:
                kwargs['label']=r'$q_\uparrow+q_\downarrow$ ($e/\AA^{2}$)'
        else:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.bwr
            if 'label' not in kwargs:
                kwargs['label']=r'$Q_\uparrow+Q_\downarrow$ ($e$)'

        super().__init__(HH.geometry, ext_geom=ext_geom, **kwargs)

        # Compute total charge on each site
        if not isinstance(spin, list):
            spin = [spin]

        # Sum over all orbitals
        chg = np.zeros((HH.geometry.na))
        for ia, io in HH.geometry.iter_orbitals(local=False):
            chg[ia] += HH.n[0, io] + HH.n[1, io]

        if realspace:
            if 'shape' not in kwargs:
                kwargs['shape'] = [100,100,1]
            if 'z' not in kwargs:
                raise ValueError('z coordinate needs to be passed to slice the real space grid')

            if 'vmin' not in kwargs:
                kwargs['vmin'] = 0

            xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

            if 'sc' not in kwargs:
                if 'z' in kwargs:
                    origin = [xmin, ymin, -kwargs['z']]
                else:
                    origin = [xmin, ymin, np.amin(HH.geometry.xyz[:,2])]

                kwargs['sc'] = sisl.SuperCell([xmax-xmin, ymax-ymin, 1000], origin=origin)

            if 'axis' not in kwargs:
                kwargs['axis'] = 2

            grid = real_space_grid(self.geometry, kwargs['sc'], chg, kwargs['shape'], mode='charge')

            # Slice it to obtain a 2D grid
            slice_grid = grid.swapaxes(kwargs['axis'], 2).grid[:, :, 0].T.real

            self.__realspace__(slice_grid, **kwargs)

        else:
            self.__orbitals__(chg, **kwargs)


class ChargeDifference(GeometryPlot):
    r""" Plot the total charge compared to the neutral atom charge in each site for the HH object

    :math:`(\langle n_{\uparrow}\rangle+\langle n_{\downarrow}\rangle)- (Z - 5)`
    Where :math:`Z` is the atomic number

    Parameters
    ----------
    HH: hubbard.HubbardHamiltonian
        mean-field Hubbard Hamiltonian
    ext_geom: sisl.Geometry, optional
        usually sp2 carbon-like systems are saturated with Hydrogen atoms, that don't play
        a role in the Hubbard Hamiltonian. If ext_geom is passed it plots the full sp2 carbon system
        otherwise it only uses the geometry associated to the `hubbard.HubbardHamiltonian` (carbon backbone)
    realspace: bool, optional
        if True it plots the charge difference in a realspace grid. In other case it will be plotted as Mulliken
        In this case the `z` kwarg needs to be passed to slice the real space grid at the desired z coordinate
    """

    def __init__(self, HH, ext_geom=None, realspace=False, **kwargs):

        # Set default kwargs
        if realspace:
            if 'facecolor' not in kwargs:
                kwargs['facecolor'] = 'None'
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'Greys'
            if 'label' not in kwargs:
                kwargs['label']=r'$q_\uparrow+q_\downarrow-q_\mathrm{NA}$ ($e/\AA^{2}$)'
        else:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.bwr
            if 'label' not in kwargs:
                kwargs['label']=r'$Q_\uparrow+Q_\downarrow-Q_\mathrm{NA}$ ($e$)'

        super().__init__(HH.geometry, ext_geom=ext_geom, **kwargs)

        # Compute total charge on each site, subtract neutral atom charge
        chg = np.zeros((HH.geometry.na))
        q = np.zeros_like(chg)
        for ia, io in HH.geometry.iter_orbitals(local=False):
            q[ia] = HH.geometry.atoms[ia].Z-5
            chg[ia] += HH.n[0,io] + HH.n[1,io]

        chg -= q

        if realspace:
            if 'shape' not in kwargs:
                kwargs['shape'] = [100,100,1]

            if 'z' not in kwargs:
                raise ValueError('z coordinate needs to be passed to slice the real space grid')


            if 'vmin' not in kwargs:
                kwargs['vmin'] = 0

            xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

            if 'sc' not in kwargs:
                if 'z' in kwargs:
                    origin = [xmin, ymin, -kwargs['z']]
                else:
                    origin = [xmin, ymin, np.amin(HH.geometry.xyz[:,2])]

                kwargs['sc'] = sisl.SuperCell([xmax-xmin, ymax-ymin, 1000], origin=origin)

            if 'axis' not in kwargs:
                kwargs['axis'] = 2

            grid = real_space_grid(self.geometry, kwargs['sc'], chg, kwargs['shape'], mode='charge')

            # Slice it to obtain a 2D grid
            slice_grid = grid.swapaxes(kwargs['axis'],2).grid[:, :, 0].T.real

            self.__realspace__(slice_grid, **kwargs)

        else:
            # Default symmetric colorscale
            if 'vmax' not in kwargs:
                kwargs['vmax'] = np.amax(np.abs(chg))
            if 'vmin' not in kwargs:
                kwargs['vmin'] = -kwargs['vmax']
            self.__orbitals__(chg, **kwargs)


class SpinPolarization(GeometryPlot):
    r""" Plot charge difference between up and down channels :math:`\langle n_{\uparrow}\rangle-\langle n_{\downarrow}\rangle`

    Parameters
    ----------
    HH: hubbard.HubbardHamiltonian
        mean-field Hubbard Hamiltonian
    ext_geom: sisl.Geometry, optional
        usually sp2 carbon-like systems are saturated with Hydrogen atoms, that don't play
        a role in the Hubbard Hamiltonian. If ext_geom is passed it plots the full sp2 carbon system
        otherwise it only uses the geometry associated to the `hubbard.HubbardHamiltonian` (carbon backbone)
    realspace: bool, optional
        if True it plots the spin polarization in a realspace grid. In other case it will be plotted as Mulliken populations
        In this case the `z` kwarg needs to be passed to slice the real space grid at the desired z coordinate
    """

    def __init__(self, HH, ext_geom=None, realspace=False, **kwargs):

        # Set default kwargs
        if realspace:
            if 'facecolor' not in kwargs:
                kwargs['facecolor'] = 'None'
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'Greys'
            if 'label' not in kwargs:
                kwargs['label']=r'$q_\uparrow-q_\downarrow$ ($e/\AA^{2}$)'
        else:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.bwr
            if 'label' not in kwargs:
                kwargs['label']=r'$Q_\uparrow-Q_\downarrow$ ($e$)'

        super().__init__(HH.geometry, ext_geom=ext_geom, **kwargs)

        # Sum over all orbitals
        chg = np.zeros((HH.geometry.na))
        for ia, io in HH.geometry.iter_orbitals(local=False):
            chg[ia] += (HH.n[0, io] - HH.n[1, io])

        if realspace:
            if 'shape' not in kwargs:
                kwargs['shape'] = [100,100,1]
            if 'z' not in kwargs:
                raise ValueError('z coordinate needs to be passed to slice the real space grid')

            if 'vmin' not in kwargs:
                kwargs['vmin'] = 0

            xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

            if 'sc' not in kwargs:
                if 'z' in kwargs:
                    origin = [xmin, ymin, -kwargs['z']]
                else:
                    origin = [xmin, ymin, np.amin(HH.geometry.xyz[:,2])]

                kwargs['sc'] = sisl.SuperCell([xmax-xmin, ymax-ymin, 1000], origin=origin)


            if 'axis' not in kwargs:
                kwargs['axis'] = 2

            grid = real_space_grid(self.geometry, kwargs['sc'], chg, kwargs['shape'], mode='charge')

            # Slice it to obtain a 2D grid
            slice_grid = grid.swapaxes(kwargs['axis'],2).grid[:, :, 0].T.real

            self.__realspace__(slice_grid, **kwargs)

        else:
            # Default symmetric colorscale
            if 'vmax' not in kwargs:
                kwargs['vmax'] = np.amax(np.abs(chg))
            if 'vmin' not in kwargs:
                kwargs['vmin'] = -kwargs['vmax']

            self.__orbitals__(chg, **kwargs)
