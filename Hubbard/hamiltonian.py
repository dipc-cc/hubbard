from __future__ import print_function
import numpy as np
import netCDF4 as NC
import sisl
import hashlib


class HubbardHamiltonian(sisl.Hamiltonian):

    def __init__(self, fn, t1=2.7, t2=0.2, t3=0.18, U=0.0, eB=3., eN=-3., Nup=0, Ndn=0,
                 nsc=[1, 1, 1], kmesh=[1, 1, 1], what=None, angle=0, v=[0, 0, 1], atom=None,
                 ncgroup='default'):
        # Save parameters
        if fn[-3:] == '.XV':
            self.fn = fn[:-3]
        elif fn[-4:] == '.xyz':
            self.fn = fn[:-4]
        # Key parameters
        self.t1 = t1 # Nearest neighbor hopping
        self.t2 = t2
        self.t3 = t3
        self.U = U # Hubbard onsite Coulomb parameter
        self.eB = eB # Boron onsite energy (relative to carbon eC=0.0)
        self.eN = eN # Nitrogen onsite energy (relative to carbon eC=0.0)
        self.Nup = Nup # Total number of up-electrons
        self.Ndn = Ndn # Total number of down-electrons
        # Read geometry etc
        ext_geom = sisl.get_sile(fn).read_geom()
        ext_geom.sc.set_nsc(nsc)
        if what:
            ext_geom = ext_geom.move(-ext_geom.center(what=what))
        ext_geom = ext_geom.rotate(angle, v, atom=atom)
        self.ext_geom = ext_geom # Keep the extended/complete geometry
        # Determine pz sites
        aux = []
        sp3 = []
        Hsp3 = []
        for ia in ext_geom:
            if ext_geom.atoms[ia].Z not in [5, 6, 7]:
                aux.append(ia)
            idx = ext_geom.close(ia, R=[0.1, 1.6])
            if len(idx[1])==4: # Search for atoms with 4 neighbors
                if ext_geom.atoms[ia].Z == 6:
                    sp3.append(ia)
                [Hsp3.append(i) for i in idx[1] if ext_geom.atoms[i].Z == 1]
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
        print(' ... B-atoms at sites', np.where(pi_geom.atoms.Z == 5)[0])
        print(' ... N-atoms at sites', np.where(pi_geom.atoms.Z == 7)[0])
        print(' ... sp3 atoms at sites', sp3)
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
        sisl.Hamiltonian.__init__(self, pi_geom, dim=2)
        self.init_hamiltonian_elements()
        # Initialize data file
        self.init_nc(self.fn+'.nc', ncgroup=ncgroup)
        # Try reading from file or use random density
        self.read(ncgroup)
        self.iterate(mix=0) # Determine midgap energy without changing densities

    def init_hamiltonian_elements(self):
        # Radii defining 1st, 2nd, and 3rd neighbors
        R = [0.1, 1.6, 2.6, 3.1]
        # Build hamiltonian for backbone
        g = self.geom
        for ia in g:
            idx = g.close(ia, R=R)
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

    def update_hamiltonian(self):
        # Update spin Hamiltonian
        g = self.geom
        for ia in g:
            # charge on neutral atom:
            n0 = g.atoms[ia].Z-5
            if g.atoms[ia].Z == 5:
                e0 = self.eB
            elif g.atoms[ia].Z == 7:
                e0 = self.eN
            else:
                e0 = 0. # onsite for C site
            self.H[ia, ia, 0] = e0 + self.U*(self.ndn[ia]-n0)
            self.H[ia, ia, 1] = e0 + self.U*(self.nup[ia]-n0)

    def random_density(self):
        print('Setting random density')
        self.nup = np.random.rand(self.sites)
        self.ndn = np.random.rand(self.sites)
        self.normalize_charge()

    def normalize_charge(self):
        """ Ensure the total up/down charge in pi-network equals Nup/Ndn """
        self.nup = self.nup/np.sum(self.nup)*self.Nup
        self.ndn = self.ndn/np.sum(self.ndn)*self.Ndn
        print('Normalized charge distributions to Nup=%i, Ndn=%i'%(self.Nup, self.Ndn))

    def set_polarization(self, up, dn=[]):
        """ Maximize spin polarization on specific atomic sites.
        Optionally, sites with down-polarization can be specified
        """
        print('Setting up-polarization for sites', up)
        for ia in up:
            self.nup[ia] = 1.
            self.ndn[ia] = 0.
        if len(dn) > 0:
            print('Setting down-polarization for sites', dn)
            for ia in dn:
                self.nup[ia] = 0.
                self.ndn[ia] = 1.
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
            niup += np.sum(np.absolute(evec_up[:, :int(Nup)])**2, axis=1).real
            nidn += np.sum(np.absolute(evec_dn[:, :int(Ndn)])**2, axis=1).real
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
        self.Etot = np.sum(ev_up[:int(Nup)])+np.sum(ev_dn[:int(Ndn)])-self.U*np.sum(nup*ndn)
        return dn, self.Etot

    def converge(self, tol=1e-10, steps=100, save=False):
        """ Iterate Hamiltonian towards a specified tolerance criterion """
        print('Iterating towards self-consistency...')
        dn = 1.0
        i = 0
        while dn > tol:
            i += 1
            if dn > 0.1:
                # precondition when density change is relatively large
                dn, Etot = self.iterate(mix=.1)
            else:
                dn, Etot = self.iterate(mix=1)
            # Print some info from time to time
            if i%steps == 0:
                print('   %i iterations completed'%i)
                # Save density to netcdf?
                if save:
                    self.save()
        print('   found solution in %i iterations'%i)
        return dn, self.Etot

    def calc_orbital_charge_overlaps(self, k=[0, 0, 0], spin=0):
        ev, evec = self.eigh(k=k, eigvals_only=False, spin=spin)
        # Compute orbital charge overlaps
        L = np.einsum('ia,ia,ib,ib->ab', evec, evec, evec, evec).real
        return ev, L

    def init_nc(self, fn, ncgroup):
        try:
            self.ncf = NC.Dataset(fn, 'a')
            print('Appending to', fn)
        except:
            print('Initiating', fn)
            self.ncf = NC.Dataset(fn, 'w')
        self.init_ncgrp(ncgroup)

    def init_ncgrp(self, ncgroup):
        if ncgroup not in self.ncf.groups:
            # create croup
            self.ncf.createGroup(ncgroup)
            self.ncf[ncgroup].createDimension('unl', None)
            self.ncf[ncgroup].createDimension('spin', 2)
            self.ncf[ncgroup].createDimension('sites', len(self.geom))
            self.ncf[ncgroup].createVariable('hash', 'i8', ('unl',))
            self.ncf[ncgroup].createVariable('U', 'f8', ('unl',))
            self.ncf[ncgroup].createVariable('Nup', 'i4', ('unl',))
            self.ncf[ncgroup].createVariable('Ndn', 'i4', ('unl',))
            self.ncf[ncgroup].createVariable('Density', 'f8', ('unl', 'spin', 'sites'))
            self.ncf[ncgroup].createVariable('Etot', 'f8', ('unl',))
            self.ncf.sync()

    def gethash(self):
        s = ''
        s += 't1=%.2f '%self.t1
        s += 't2=%.2f '%self.t2
        s += 't3=%.2f '%self.t3
        s += 'U=%.2f '%self.U
        s += 'eB=%.2f '%self.eB
        s += 'eN=%.2f '%self.eN
        s += 'Nup=%.2f '%self.Nup
        s += 'Ndn=%.2f '%self.Ndn
        myhash = int(hashlib.md5(s).hexdigest()[:7], 16)
        return myhash, s

    def save(self, ncgroup='default'):
        myhash, s = self.gethash()
        self.init_ncgrp(ncgroup)
        i = np.where(self.ncf[ncgroup]['hash'][:] == myhash)[0]
        if len(i) == 0:
            i = len(self.ncf[ncgroup]['hash'][:])
        else:
            i = i[0]
        self.ncf[ncgroup]['hash'][i] = myhash
        self.ncf[ncgroup]['U'][i] = self.U
        self.ncf[ncgroup]['Nup'][i] = self.Nup
        self.ncf[ncgroup]['Ndn'][i] = self.Ndn
        self.ncf[ncgroup]['Density'][i, 0] = self.nup
        self.ncf[ncgroup]['Density'][i, 1] = self.ndn
        self.ncf[ncgroup]['Etot'][i] = self.Etot
        self.ncf.sync()
        print('Wrote (U,Nup,Ndn)=(%.2f,%i,%i) data to %s.nc{%s}'%(self.U, self.Nup, self.Ndn, self.fn, ncgroup))

    def read(self, ncgroup=None):
        myhash, s = self.gethash()
        if ncgroup == None:
            # Lookup if hash exists in any group
            for grp in self.ncf.groups:
                ncgroup = grp
                i = np.where(self.ncf[grp]['hash'][:] == myhash)[0]
                if len(i) > 0:
                    break
        i = np.where(self.ncf[ncgroup]['hash'][:] == myhash)[0]
        if len(i) == 0:
            print('Hash not found:')
            print('...', s)
            self.random_density()
        else:
            print('Found:')
            print('... %s in %s.nc{%s}' % (s, self.fn, ncgroup))
            i = i[0]
            self.U = self.ncf[ncgroup]['U'][i]
            self.nup = self.ncf[ncgroup]['Density'][i][0]
            self.ndn = self.ncf[ncgroup]['Density'][i][1]
            self.Etot = self.ncf[ncgroup]['Etot'][i]
        self.update_hamiltonian()


