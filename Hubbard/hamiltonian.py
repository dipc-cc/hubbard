"""

:mod:`Hubbard.hamiltonian`
==========================

Function for the meanfield Hubbard Hamiltonian

.. currentmodule:: Hubbard.hamiltonian

"""

from __future__ import print_function
import numpy as np
import sisl


class HubbardHamiltonian(sisl.Hamiltonian):
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
    U : float, optional
      on-site Coulomb repulsion
    eB : float, optional
      on-site energy for Boron atoms
    eN : float, optional
      on-site energy for Nitrogen atoms  
    kmesh : array_like, optional
      number of k-points in the interval [0, 1] (units
      of [pi/a]) along each direction in which the Hamiltonian 
      will be evaluated

    """

    def __init__(self, ext_geom, t1=2.7, t2=0.2, t3=0.18, U=0.0, eB=3., eN=-3., Nup=0, Ndn=0,
                  kmesh=[1, 1, 1], s0=1.0, s1=0, s2=0, s3=0):
        """ Initialize HubbardHamiltonian """
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
        self.U = U # Hubbard onsite Coulomb parameter
        self.eB = eB # Boron onsite energy (relative to carbon eC=0.0)
        self.eN = eN # Nitrogen onsite energy (relative to carbon eC=0.0)
        self.Nup = Nup # Total number of up-electrons
        self.Ndn = Ndn # Total number of down-electrons
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
        print('Found %i B-atoms, %i C-atoms, %i N-atoms' %(nB, nC, nN))
        print('Neutral system corresponds to a total of %i electrons' %ntot)
        # Use default (low-spin) filling?
        if Ndn <= 0:
            self.Ndn = int(ntot/2)
        if Nup <= 0:
            self.Nup = int(ntot-self.Ndn)
        # Generate kmesh
        [nx, ny, nz] = kmesh
        self.kmesh = []
        for kx in np.arange(0, 1, 1./nx):
            for ky in np.arange(0, 1, 1./ny):
                for kz in np.arange(0, 1, 1./nz):
                    self.kmesh.append([kx, ky, kz])
        # Construct Hamiltonians
        sisl.Hamiltonian.__init__(self, pi_geom, orthogonal=orthogonal, dim=2)
        # Generate Monkhorst-Pack
        self.mp = sisl.MonkhorstPack(self, kmesh)
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
        g = self.geom
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
        # Determine midgap with U=0
        ev = self.H.eigh(spin=0)
        N = max(self.Nup, self.Ndn)
        HOMO = ev[N-1]
        LUMO = ev[N]
        self.midgap = (LUMO+HOMO)/2

    def update_hamiltonian(self):
        # Update spin Hamiltonian
        g = self.geom

        # Faster to loop individual species
        E = np.empty([len(g), 2])
        E[:, 0] = self.U * self.ndn
        E[:, 1] = self.U * self.nup
        for atom, ias in g.atoms.iter(True):
            # charge on neutral atom
            n0 = atom.Z - 5
            if atom.Z == 5:
                e0 = self.eB
            elif atom.Z == 7:
                e0 = self.eN
            else:
                e0 = 0.

            # Faster to do it for more than one element at a time
            E[ias, :] += e0 - self.U * n0

            for ia in ias:
                self.H[ia, ia, [0, 1]] = E[ia]

    def random_density(self):
        """ Initialize spin polarization  with random density """
        print('Setting random density')
        self.nup = np.random.rand(self.sites)
        self.ndn = np.random.rand(self.sites)
        self.normalize_charge()

    def normalize_charge(self):
        """ Ensure the total up/down charge in pi-network equals Nup/Ndn """
        self.nup = self.nup / self.nup.sum() * self.Nup
        self.ndn = self.ndn / self.ndn.sum() * self.Ndn
        print('Normalized charge distributions to Nup=%i, Ndn=%i'%(self.Nup, self.Ndn))

    def set_polarization(self, up, dn=[]):
        """ Maximize spin polarization on specific atomic sites.
        Optionally, sites with down-polarization can be specified

        Parameters
        ----------
        up : array_like
            atomic sites where the spin-up density is going to be maximized
        dn : array_like, optional
            atomic sites where the spin-down density is going to be maximized
        """
        print('Setting up-polarization for sites', up)
        self.nup[up] = 1.
        self.ndn[up] = 0.
        if len(dn) > 0:
            print('Setting down-polarization for sites', dn)
            self.nup[dn] = 0.
            self.ndn[dn] = 1.
        self.normalize_charge()

    def iterate(self, mix=1.0):
        nup = self.nup
        ndn = self.ndn
        Nup = self.Nup
        Ndn = self.Ndn
        # Solve eigenvalue problems
        niup = 0*nup
        nidn = 0*ndn
        HOMO = -1e10
        LUMO = 1e10
        for ik, k in enumerate(self.kmesh):
            ev_up, evec_up = self.eigh(k=k, eigvals_only=False, spin=0)
            ev_dn, evec_dn = self.eigh(k=k, eigvals_only=False, spin=1)
            # Compute new occupations
            if self.H.orthogonal:
                niup += np.sum(np.absolute(evec_up[:, :int(Nup)])**2, axis=1).real
                nidn += np.sum(np.absolute(evec_dn[:, :int(Ndn)])**2, axis=1).real
            else:
                S = self.H.Sk().todense()
                Dup = np.einsum('ia,ja->ij', evec_up[:, :int(Nup)], evec_up[:, :int(Nup)]).real
                Ddn = np.einsum('ia,ja->ij', evec_dn[:, :int(Ndn)], evec_dn[:, :int(Ndn)]).real
                niup += 0.5*(np.einsum('ij,ij->i', Dup, S) + np.einsum('ji,ji->i', Dup.T, S.T))
                nidn += 0.5*(np.einsum('ij,ij->i', Ddn, S) + np.einsum('ji,ji->i', Ddn.T, S.T))
            HOMO = max(HOMO, ev_up[self.Nup-1], ev_dn[self.Ndn-1])
            LUMO = min(LUMO, ev_up[self.Nup], ev_dn[self.Ndn])
        niup = niup/len(self.kmesh)
        nidn = nidn/len(self.kmesh)
        # Determine midgap energy reference
        self.midgap = (LUMO+HOMO)/2
        # Measure of density change
        dn = np.sum(abs(nup-niup)+abs(ndn-nidn))
        # Update occupations
        self.nup = mix*niup+(1.-mix)*nup
        self.ndn = mix*nidn+(1.-mix)*ndn
        # Update spin hamiltonian
        self.update_hamiltonian()
        # Compute total energy
        self.Etot = np.sum(ev_up[:int(Nup)]) + np.sum(ev_dn[:int(Ndn)]) - self.U*np.sum(self.nup*self.ndn)
        return dn

    def iterate2(self, mix=1.0):
        # Create short-hands
        nup = self.nup
        ndn = self.ndn
        Nup = int(self.Nup)
        Ndn = int(self.Ndn)

        # Initialize HOMO/LUMO variables
        HOMO = -1e10
        LUMO = 1e10

        # Initialize new occupations and total energy with Hubbard U
        ni_up = np.zeros(nup.shape)
        ni_dn = np.zeros(ndn.shape)
        Etot = 0

        # Solve eigenvalue problems
        def calc_occ(k, weight, HOMO, LUMO):
            """ My wrap function for calculating occupations """
            es_up = self.eigenstate(k, spin=0)
            es_dn = self.eigenstate(k, spin=1)

            # Update HOMO, LUMO
            HOMO = max(HOMO, es_up.eig[Nup-1], es_dn.eig[Ndn-1])
            LUMO = min(LUMO, es_up.eig[Nup], es_dn.eig[Ndn])
            
            es_up = es_up.sub(range(Nup))
            es_dn = es_dn.sub(range(Ndn))

            ni_up = es_up.norm(False).sum(0) * weight
            ni_dn = es_dn.norm(False).sum(0) * weight

            # Calculate total energy
            Etot = (es_up.eig.sum() + es_dn.eig.sum()) * weight
            # Return values
            return HOMO, LUMO, ni_up, ni_dn, Etot

        # Loop k-points and weights
        for w, k in zip(self.mp.weight, self.mp.k):
            HOMO, LUMO, up, dn, etot = calc_occ(k, w, HOMO, LUMO)
            ni_up += up
            ni_dn += dn
            Etot += etot

        # Determine midgap energy reference
        self.midgap = (LUMO + HOMO) / 2
        
        # Measure of density change
        dn = (np.absolute(nup - ni_up) + np.absolute(ndn - ni_dn)).sum()

        # Update occupations on sites with mixing
        self.nup = mix * ni_up + (1. - mix) * nup
        self.ndn = mix * ni_dn + (1. - mix) * ndn
        
        # Update spin hamiltonian
        self.update_hamiltonian()

        # Store total energy
        self.Etot = Etot - self.U * (self.nup * self.ndn).sum()

        return dn

    def iterate3(self, mix=1.0, q_up=None, q_dn=None):
        # Create short-hands
        nup = self.nup
        ndn = self.ndn
        if q_up is None:
            q_up = self.Nup
        if q_dn is None:
            q_dn = self.Ndn

        # To do metallic systems one should use this thing to
        # calculate the fermi-level:
        kT = 0.00001
        # Create fermi-level determination distribution
        dist = sisl.get_distribution('fermi_dirac', smearing=kT)
        Ef = self.fermi_level(self.mp, q=[q_up, q_dn], distribution=dist)
        dist_up = sisl.get_distribution('fermi_dirac', smearing=kT, x0=Ef[0])
        dist_dn = sisl.get_distribution('fermi_dirac', smearing=kT, x0=Ef[1])

        # Initialize new occupations and total energy with Hubbard U
        ni_up = np.zeros(nup.shape)
        ni_dn = np.zeros(ndn.shape)
        Etot = 0

        # Solve eigenvalue problems
        def calc_occ(k, weight):
            """ My wrap function for calculating occupations """
            es_up = self.eigenstate(k, spin=0)
            es_dn = self.eigenstate(k, spin=1)

            # Reduce to occupied stuff
            occ_up = es_up.occupation(dist_up).reshape(-1, 1) * weight
            ni_up = (es_up.norm(False) * occ_up).sum(0)
            occ_dn = es_dn.occupation(dist_dn).reshape(-1, 1) * weight
            ni_dn = (es_dn.norm(False) * occ_dn).sum(0)
            Etot = (es_up.eig * occ_up.ravel()).sum() + (es_dn.eig * occ_dn.ravel()).sum()

            # Return values
            return ni_up, ni_dn, Etot

        # Loop k-points and weights
        for w, k in zip(self.mp.weight, self.mp.k):
            up, dn, etot = calc_occ(k, w)
            ni_up += up
            ni_dn += dn
            Etot += etot

        # Determine midgap energy reference (or simply the fermi-level)
        self.midgap = Ef.sum() / 2
        
        # Measure of density change
        dn = (np.absolute(nup - ni_up) + np.absolute(ndn - ni_dn)).sum()

        # Update occupations on sites with mixing
        self.nup = mix * ni_up + (1. - mix) * nup
        self.ndn = mix * ni_dn + (1. - mix) * ndn
        
        # Update spin hamiltonian
        self.update_hamiltonian()

        # Store total energy
        self.Etot = Etot - self.U * (self.nup * self.ndn).sum()

        return dn

    def converge(self, tol=1e-10, steps=100, mix=1.0, premix=0.1, method=0):
        """ Iterate Hamiltonian towards a specified tolerance criterion """
        print('Iterating towards self-consistency...')
        if method == 2:
            iterate_ = self.iterate2
        elif method == 3:
            iterate_ = self.iterate3
        else:
            iterate_ = self.iterate

        dn = 1.0
        i = 0
        while dn > tol:
            i += 1
            if dn > 0.1:
                # precondition when density change is relatively large
                dn = iterate_(mix=premix)
            else:
                dn = iterate_(mix=mix)
            # Print some info from time to time
            if i%steps == 0:
                print('   %i iterations completed:'%i, dn, self.Etot)

        print('   found solution in %i iterations'%i)
        return dn

    def calc_orbital_charge_overlaps(self, k=[0, 0, 0], spin=0):
        ev, evec = self.eigh(k=k, eigvals_only=False, spin=spin)
        # Compute orbital charge overlaps
        L = np.einsum('ia,ia,ib,ib->ab', evec, evec, evec, evec).real
        return ev, L


    def get_Zak_phase(self, Nx=51, sub='filled', eigvals=False):
        """ Compute Zak phase for 1D systems oriented along the x-axis.
        Keep in mind that the current implementation does not handle correctly band intersections.
        Meaningful Zak phases can thus only be computed for the non-crossing bands.
        """
        # Discretize kx over [0.0, 1.0[ in Nx-1 segments (1BZ)
        def func(sc, frac):
            return [frac, 0, 0]
        bz = sisl.BrillouinZone(self).parametrize(self, func, Nx)
        if sub == 'filled':
            # Sum up over all occupied bands:
            sub = range(self.Nup)
        return sisl.electron.berry_phase(bz, sub=sub, eigvals=eigvals, method='zak')


    def get_bond_order(self, format='csr'):
        """ Compute Huckel bond order

        Parameters
        ----------
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`) or `numpy.matrix` (`'dense'`).

        Returns
        -------
        object : the Huckel bond-order matrix
        """
        g = self.geom
        BO = sisl.Hamiltonian(g)
        R = [0.1, 1.6]
        for ik, k in enumerate(self.kmesh):
            # spin-up first
            ev, evec = self.eigh(k=k, eigvals_only=False, spin=0)
            ev -= self.midgap
            idx = np.where(ev < 0.)[0]
            bo = np.dot(np.conj(evec[:, idx]), evec[:, idx].T)
            # add spin-down
            ev, evec = self.eigh(k=k, eigvals_only=False, spin=1)
            ev -= self.midgap
            idx = np.where(ev < 0.)[0]
            bo += np.dot(np.conj(evec[:, idx]), evec[:, idx].T)
            for ix in (-1, 0, 1):
                for iy in (-1, 0, 1):
                    for iz in (-1, 0, 1):
                        r = (ix, iy, iz)
                        phase = np.exp(-2.j*np.pi*np.dot(k, r))
                        for ia in g:
                            for ja in g.close_sc(ia, R=R, isc=r)[1]:
                                bor = bo[ia, ja]*phase
                                BO[ia, ja] += bor.real
        BO /= len(self.kmesh)
        # Add sigma bond at the end
        for ia in g:
            idx = g.close(ia, R=R)
            BO[ia, idx[1]] += 1.
        return BO.Hk(format=format) # Fold to Gamma

    def spin_contamination(self):
        """
        Obtains the spin contamination after the MFH calculation
        Ref. Chemical Physics Letters. 183 (5): 423–431.
        
        This function works for non-periodic systems only.
        """
        # Define Nalpha and Nbeta, where Nalpha >= Nbeta 
        Nalpha = max(self.Nup, self.Ndn)
        Nbeta = min(self.Nup, self.Ndn)
        
        # Exact Total Spin expected value (< S² >)
        S = .5*(Nalpha - Nbeta) * ( (Nalpha - Nbeta)*.5 + 1)

        # Extract eigenvalues and eigenvectors of spin-up and spin-dn electrons
        ev_up, evec_up = self.eigh(eigvals_only=False, spin=0)
        ev_dn, evec_dn = self.eigh(eigvals_only=False, spin=1)

        # No need to tell which matrix of eigenstates correspond to alpha or beta, 
        # the sisl function spin_squared already takes this into account
        s2alpha, s2beta = sisl.electron.spin_squared(evec_up[:, :self.Nup].T, evec_dn[:, :self.Ndn].T)
        
        # Spin contamination 
        S_MFH = S + Nbeta - s2beta.sum()

        return S, S_MFH

    def band_sym(self, eigenstate, diag=True):
        '''
        Obtains the parity of vector(s) with respect to the rotation of its parent geometry by 180 degrees
        '''
        geom0 = self.geom
        geom180 = geom0.rotate(180, [0, 0, 1], geom0.center())
        sites180 = []
        for ia in geom180:
            for ib in geom0:
                if np.allclose(geom0.xyz[ib], geom180.xyz[ia]):
                    sites180.append(ib)
        if isinstance(eigenstate, sisl.physics.electron.EigenstateElectron):
            # In eigenstate instance dimensions are: (En, sites)
            v1 = np.conjugate(eigenstate.state)
            v2 = eigenstate.state[:, sites180]
        else:
            # Transpose to have dimensions (En, sites)
            if len(eigenstate.shape) == 1:
                eigenstate = eigenstate.reshape(1, eigenstate.shape[0]) 
            else:
                eigenstate = eigenstate.T
            v1 = np.conjugate(eigenstate)
            v2 = eigenstate[:, sites180]

        if diag:
            sym = (v1 * v2).sum(axis=1)
        else:
            sym = np.dot(v1, v2.T)
        return sym

    def DOS(self, egrid, eta=1e-3, spin=[0,1]):
        """
        Obtains the Density Of States of the system convoluted with a Lorentzian function

        Parameters
        ----------
        egrid: float or array_like
            Energy grid at which the DOS will be calculated.
        eta: float
            Smearing parameter
        spin: integer or list of integers
            If spin=0(1) it calculates the DOS for up (down) electrons in the system.
            If spin is not specified it returns DOS_up + DOS_dn.
        """

        # Check if egrid is numpy.ndarray
        if not isinstance(egrid, (np.ndarray)):
            egrid = np.array(egrid)
        
        # Obtain DOS
        DOS = 0
        for ispin in spin:
            ev, evec = self.eigh(eigvals_only=False, spin=ispin)
            ev -= self.midgap

            id1 = np.ones(ev.shape,np.float)
            id2 = np.ones(egrid.shape,np.float)
            dE = np.outer(id1,egrid)-np.outer(ev,id2)
            LOR = 2*eta/(dE**2+eta**2)
            DOS += np.einsum('ai,ai,ij->aj',evec,evec,LOR)/(2*np.pi)
        
        return DOS
