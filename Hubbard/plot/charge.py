import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
import sisl
import numpy as np

__all__ = ['Charge', 'ChargeDifference', 'SpinPolarization']


class Charge(GeometryPlot):
    r""" Plot the total charge for the HubbardHamiltonian object

    :math:`\langle n_{\uparrow}\rangle+\langle n_{\downarrow}\rangle`

    or the charge for a particular spin-channel (:math:`\langle n_{\uparrow}\rangle`)
    by specifying the `spin` index.

    Notes
    -----
    If the `realspace` kwarg is passed then it will plot it in a realspace grid.
    In other case it will be plotted as Mulliken populations.
    """

    def __init__(self, HubbardHamiltonian, ext_geom=None, spin=[0, 1], realspace=False, **kwargs):
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

        super().__init__(HubbardHamiltonian.geometry, ext_geom=ext_geom, **kwargs)

        # Compute total charge on each site
        if not isinstance(spin, list):
            spin = [spin]

        charge = HubbardHamiltonian.dm[spin].sum(axis=0)

        if realspace:
            self.__realspace__(charge, density=True, **kwargs)

        else:
            self.__orbitals__(charge, **kwargs)


class ChargeDifference(GeometryPlot):
    r""" Plot the total charge compared to the neutral atom charge in each site for the HubbardHamiltonian object

    :math:`(\langle n_{\uparrow}\rangle+\langle n_{\downarrow}\rangle)- (Z - 5)`
    Where :math:`Z` is the atomic number

    Notes
    -----
    If the `realspace` kwarg is passed then it will plot it in a realspace grid.
    In other case it will be plotted as Mulliken populations.
    """

    def __init__(self, HubbardHamiltonian, ext_geom=None, realspace=False, **kwargs):

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

        super().__init__(HubbardHamiltonian.geometry, ext_geom=ext_geom, **kwargs)

        # Compute total charge on each site, subtract neutral atom charge
        charge = HubbardHamiltonian.dm.sum(0)
        for ia in HubbardHamiltonian.geometry:
            charge[ia] -= HubbardHamiltonian.geometry.atoms[ia].Z-5

        if realspace:
            self.__realspace__(charge, density=True, **kwargs)

        else:
            self.__orbitals__(charge, **kwargs)


class SpinPolarization(GeometryPlot):
    r""" Plot charge difference between up and down channels :math:`\langle n_{\uparrow}\rangle-\langle n_{\downarrow}\rangle`

    Notes
    -----
    If the `realspace` kwarg is passed then it will plot it in a realspace grid.
    In other case it will be plotted as Mulliken populations.
    """

    def __init__(self, HubbardHamiltonian, ext_geom=None, realspace=False, **kwargs):

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

        super().__init__(HubbardHamiltonian.geometry, ext_geom=ext_geom, **kwargs)

        # Compute charge difference between up and down channels
        charge = np.diff(HubbardHamiltonian.dm[[1, 0]], axis=0).ravel()

        if realspace:
            self.__realspace__(charge, density=True, **kwargs)

        else:
            self.__orbitals__(charge, **kwargs)
