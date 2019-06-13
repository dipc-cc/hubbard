"""

:mod:`Hubbard.hamiltonian`
==========================

Function for the meanfield Hubbard Hamiltonian

.. currentmodule:: Hubbard.sp2

"""

from __future__ import print_function
import numpy as np
import netCDF4 as NC
import sisl
import hashlib


class sp2(object):
    """ sisl-type object

    Parameters:
    -----------
    ext_geom : Geometry (sisl) object
        complete geometry that hosts also the SuperCell
        information (for instance the direction of periodicity, etc.)
    t1 : float, optional
      nearest neighbor hopping matrix element
    t2 : float, optional
      second nearest neighbor hopping matrix element
    t3 : float, optional
      third nearest neigbor hopping matrix element
    eB : float, optional
      on-site energy for Boron atoms
    eN : float, optional
      on-site energy for Nitrogen atoms  
    kmesh : array_like, optional
      number of k-points in the interval [0, 1] (units
      of [pi/a]) along each direction in which the Hamiltonian 
      will be evaluated
    dim : int, optional
        Dimension of Hamiltonian. If dim=2(1) the resulting Hamiltonian 
        will (not) be spin-polarized.
    """

    def __init__(self, ext_geom, t1=2.7, t2=0.2, t3=0.18, eB=3., eN=-3.,
                  kmesh=[1, 1, 1], s0=1.0, s1=0, s2=0, s3=0, dim=1, N=0):
        """ Initialize Tight Binding Hamiltonian for sp2 Carbon systems"""
        self.ext_geom = ext_geom # Keep the extended/complete geometry
        # Key parameters
        self.t1 = t1 # Nearest neighbor hopping
        self.t2 = t2
        self.t3 = t3
        self.s0 = s0 # Self overlap matrix element
        self.s1 = s1 # Overlap matrix element between 1NN
        self.s2 = s2 # Overlap matrix element between 2NN
        self.s3 = s3 # Overlap matrix element between 3NN
        if self.s1 != 0:
            orthogonal = False
        else:
            orthogonal = True
        self.eB = eB # Boron onsite energy (relative to carbon eC=0.0)
        self.eN = eN # Nitrogen onsite energy (relative to carbon eC=0.0)
        # Determine pz sites
        aux = []
        sp3 = []
        for ia in ext_geom:
            if ext_geom.atoms[ia].Z not in [5, 6, 7]:
                aux.append(ia)
            idx = ext_geom.close(ia, R=[0.1, 1.6])
            if len(idx[1])==4: # Search for atoms with 4 neighbors
                if ext_geom.atoms[ia].Z == 6:
                    sp3.append(ia)
        # Remove all sites not carbon-type
        pi_geom = ext_geom.remove(aux+sp3)
        self.sites = len(pi_geom)
        print('Found %i pz sites' %self.sites)
        # Set pz orbital for each pz site
        r = np.linspace(0, 1.6, 700)
        func = 5 * np.exp(-r * 5)
        pz = sisl.SphericalOrbital(1, (r, func))
        for ia in pi_geom:
            pi_geom.atom[ia].orbital[0] = pz
        # Count number of pi-electrons:
        nB = len(np.where(pi_geom.atoms.Z == 5)[0])
        nC = len(np.where(pi_geom.atoms.Z == 6)[0])
        nN = len(np.where(pi_geom.atoms.Z == 7)[0])
        ntot = 0*nB+1*nC+2*nN
        if N <= 0:
            self.N = int(ntot/2)
        print('Found %i B-atoms, %i C-atoms, %i N-atoms' %(nB, nC, nN))
        print('Neutral system corresponds to a total of %i electrons' %ntot)
        # Construct Hamiltonians
        self.H = sisl.Hamiltonian(pi_geom, orthogonal=orthogonal, dim=dim)
        # Initialize elements
        self.init_hamiltonian_elements()

    def init_hamiltonian_elements(self):
        """ Setup the initial Hamiltonian
        
        Set Hamiltonian matrix elements H_ij, where ij are pairs of atoms separated by a distance defined as:
        R = [on-site, 1NN, 2NN, 3NN]
        
        """
        # Radii defining 1st, 2nd, and 3rd neighbors
        R = [0.1, 1.6, 2.6, 3.1]
        # Build hamiltonian for backbone
        g = self.H.geom
        for ia in g:
            idx = g.close(ia, R=R)
            # NB: I found that ':' is necessary in the following lines, but I don't understand why...
            if g.atoms[ia].Z == 5:
                self.H[ia, ia, :] = self.eB # set onsite for B sites
            elif g.atoms[ia].Z == 7:
                self.H[ia, ia, :] = self.eN # set onsite for N sites
            # set hoppings
            self.H[ia, idx[1], :] = -self.t1
            if self.t2 != 0:
                self.H[ia, idx[2], :] = -self.t2
            if self.t3 != 0:
                self.H[ia, idx[3], :] = -self.t3
            if not self.H.orthogonal:
                self.H.S[ia, ia] = self.s0
                self.H.S[ia, idx[1]] = self.s1
                self.H.S[ia, idx[2]] = self.s2
                self.H.S[ia, idx[3]] = self.s3
        # Determine midgap
        ev = self.H.eigh(spin=0)
        HOMO = ev[self.N-1]
        LUMO = ev[self.N]
        self.midgap = (LUMO+HOMO)/2

    def add_Hatoms(self):
        '''
        Function to add H atoms at the edges 
        '''
        pass
