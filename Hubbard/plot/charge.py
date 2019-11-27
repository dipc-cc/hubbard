from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
import sisl
import numpy as np


class Charge(GeometryPlot):

    def __init__(self, HubbardHamiltonian, ext_geom=None, spin=[0,1], **keywords):
        # Set default keywords
        if 'realspace' in keywords:
            if 'facecolor' not in keywords:
                keywords['facecolor'] = 'None'
            if 'cmap' not in keywords:
                keywords['cmap'] = 'Greys'
            if 'label' not in keywords:
                keywords['label']=r'$q_\uparrow+q_\downarrow$ ($e/$\AA)'
        else:
            if 'cmap' not in keywords:
                keywords['cmap'] = plt.cm.bwr
            if 'label' not in keywords:
                keywords['label']=r'$Q_\uparrow+Q_\downarrow$ ($e$)'

        GeometryPlot.__init__(self, HubbardHamiltonian.geom, ext_geom=ext_geom, **keywords)

        # Compute total charge on each site
        n = np.array([HubbardHamiltonian.nup, HubbardHamiltonian.ndn])
        if not isinstance(spin, list):
            spin = [spin]

        charge = n[spin].sum(axis=0)

        if 'realspace' in keywords:
            self.__realspace__(charge, density=True, **keywords)

        else:
            self.__orbitals__(charge, **keywords)

class ChargeDifference(GeometryPlot):

    def __init__(self, HubbardHamiltonian, ext_geom=None, **keywords):

        # Set default keywords
        if 'realspace' in keywords:
            if 'facecolor' not in keywords:
                keywords['facecolor'] = 'None'
            if 'cmap' not in keywords:
                keywords['cmap'] = 'Greys'
            if 'label' not in keywords:
                keywords['label']=r'$q_\uparrow+q_\downarrow-q_\mathrm{NA}$ ($e/$\AA)'
        else:
            if 'cmap' not in keywords:
                keywords['cmap'] = plt.cm.bwr
            if 'label' not in keywords:
                keywords['label']=r'$Q_\uparrow+Q_\downarrow-Q_\mathrm{NA}$ ($e$)'

        GeometryPlot.__init__(self, HubbardHamiltonian.geom, ext_geom=ext_geom, **keywords)

        # Compute total charge on each site, subtract neutral atom charge
        charge = HubbardHamiltonian.nup + HubbardHamiltonian.ndn
        for ia in HubbardHamiltonian.geom:
            charge[ia] -= HubbardHamiltonian.geom.atoms[ia].Z-5

        if 'realspace' in keywords:
            self.__realspace__(charge, density=True, **keywords)

        else:
            self.__orbitals__(charge, **keywords)

class SpinPolarization(GeometryPlot):

    def __init__(self, HubbardHamiltonian, ext_geom=None, **keywords):

        # Set default keywords
        if 'realspace' in keywords:
            if 'facecolor' not in keywords:
                keywords['facecolor'] = 'None'
            if 'cmap' not in keywords:
                keywords['cmap'] = 'Greys'
            if 'label' not in keywords:
                keywords['label']=r'$q_\uparrow-q_\downarrow$ ($e/$\AA)'
        else:
            if 'cmap' not in keywords:
                keywords['cmap'] = plt.cm.bwr
            if 'label' not in keywords:
                keywords['label']=r'$Q_\uparrow-Q_\downarrow$ ($e$)'

        GeometryPlot.__init__(self, HubbardHamiltonian.geom, ext_geom=ext_geom, **keywords)

        # Compute charge difference between up and down channels
        charge = HubbardHamiltonian.nup - HubbardHamiltonian.ndn
        
        if 'realspace' in keywords:
            self.__realspace__(charge, density=True, **keywords)

        else:
            self.__orbitals__(charge, **keywords)

