from __future__ import print_function
import numpy as np
import netCDF4 as NC
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors 
import sisl
import hashlib


class Hubbard(object):

    def __init__(self, fn, t1=2.7, t2=0.2, t3=0.18, U=0.0, eB=3., eN=-3., Nup=0, Ndn=0,
                 nsc=[1, 1, 1], kmesh=[1, 1, 1], what=None, angle=0, v=[0, 0, 1], atom=None, write_xyz=False):
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
        # Determine whether this is 1NN or 3NN
        if self.t3 == 0:
            self.model = '1NN'
        else:
            self.model = '3NN'
        # Read geometry etc
        self.geom = sisl.get_sile(fn).read_geom()
        self.geom.sc.set_nsc(nsc)
        if what:
            self.geom = self.geom.move(-self.geom.center(what=what))
        self.geom = self.geom.rotate(angle, v, atom=atom)
        # NB Force structure planar!!! (setting z = 0)
        self.geom.xyz[:,2] = 0
        if write_xyz:
            self.geom.write(self.fn+'.xyz')
            print('Wrote '+self.fn+'.xyz')
        # Determine pz sites
        aux = []
        for ia in self.geom:
            if self.geom.atoms[ia].Z not in [5, 6, 7]:
                aux.append(ia)
        # Remove all sites not carbon-type
        self.pi_geom = self.geom.remove(aux)
        self.sites = len(self.pi_geom)
        print('Found %i pz sites' %self.sites)
        # Set pz orbital for each pz site
        r = np.linspace(0, 1.6, 700)
        func = 5 * np.exp(-r * 5)
        pz = sisl.SphericalOrbital(1, (r, func))
        for ia in self.pi_geom:
            self.pi_geom.atom[ia].orbital[0] = pz
        # Count number of pi-electrons:
        nB = len(np.where(self.pi_geom.atoms.Z == 5)[0])
        nC = len(np.where(self.pi_geom.atoms.Z == 6)[0])
        nN = len(np.where(self.pi_geom.atoms.Z == 7)[0])
        ntot = 0*nB+1*nC+2*nN
        print('Found %i B-atoms, %i C-atoms, %i N-atoms' %(nB, nC, nN))
        print(' ... B-atoms at sites', np.where(self.pi_geom.atoms.Z == 5)[0])
        print(' ... N-atoms at sites', np.where(self.pi_geom.atoms.Z == 7)[0])
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
        self.setup_hamiltonians()
        # Initialize data file
        self.init_nc(self.fn+'.nc')
        # Try reading from file or use random density
        self.read()
        self.iterate(mix=0) # Determine midgap energy without changing densities
        # Call dumb plot to avoid font issues with first real Matplotlib call
        self.dumb_plot()

    def dumb_plot(self):
        "This function does nothing but avoids font conversion issues"
        fig = plt.figure()
        plt.rc('text', usetex=True)
        plt.close('all')

    def get_label(self):
        s = self.fn
        s += '-%s'%self.model
        s += '-U%.3i'%(self.U*100)
        return s

    def polarize(self, pol):
        print('Polarizing by', pol)
        self.Nup += int(pol)
        self.Ndn -= int(pol)
        return self.Nup, self.Ndn

    def setup_hamiltonians(self):
        # Radii defining 1st, 2nd, and 3rd neighbors
        R = [0.1, 1.6, 2.6, 3.1]
        # Build hamiltonian for backbone
        g = self.pi_geom
        self.H0 = sisl.Hamiltonian(g)
        for ia in g:
            idx = g.close(ia, R=R)
            if g.atoms[ia].Z == 5:
                # set onsite for B sites
                self.H0.H[ia, ia] = self.eB
                print('Found B site')
            # set onsite for N sites
            if g.atoms[ia].Z == 7:
                self.H0.H[ia, ia] = self.eN
                print('Found N site')
            # set hoppings
            self.H0.H[ia, idx[1]] = -self.t1
            if self.t2 != 0:
                self.H0.H[ia, idx[2]] = -self.t2
            if self.t3 != 0:
                self.H0.H[ia, idx[3]] = -self.t3
        self.Hup = self.H0.copy()
        self.Hdn = self.H0.copy()

    def update_spin_hamiltonian(self):
        # Update spin Hamiltonian
        for ia in self.pi_geom:
            # charge on neutral atom:
            n0 = self.pi_geom.atoms[ia].Z-5
            self.Hup.H[ia, ia] = self.H0.H[ia, ia] + self.U*(self.ndn[ia]-n0)
            self.Hdn.H[ia, ia] = self.H0.H[ia, ia] + self.U*(self.nup[ia]-n0)

    def random_density(self):
        print('Setting random density')
        self.nup = np.random.rand(self.sites)
        self.nup = self.nup/np.sum(self.nup)*(self.Nup)
        self.ndn = np.random.rand(self.sites)
        self.ndn = self.ndn/np.sum(self.ndn)*(self.Ndn)

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
            ev_up, evec_up = self.Hup.eigh(k=k, eigvals_only=False)
            ev_dn, evec_dn = self.Hdn.eigh(k=k, eigvals_only=False)
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
        dn = np.sum(abs(nup-niup))
        # Update occupations
        self.nup = mix*niup+(1.-mix)*nup
        self.ndn = mix*nidn+(1.-mix)*ndn
        # Update spin hamiltonian
        self.update_spin_hamiltonian()
        # Compute total energy
        self.Etot = np.sum(ev_up[:int(Nup)])+np.sum(ev_dn[:int(Ndn)])-self.U*np.sum(nup*ndn)
        return dn, self.Etot

    def get_atomic_patches(self, cmax=10, cmap=plt.cm.bwr, **keywords):
        ppi = [] # patches for pi-network
        paux = [] # patches for other auxillary atoms
        g = self.geom
        for ia in g:
            if g.atoms[ia].Z == 1: # H
                paux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.4))
            elif g.atoms[ia].Z == 5: # B
                ppi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.0))
            elif g.atoms[ia].Z == 6: # C
                ppi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
            elif g.atoms[ia].Z == 7: # N
                ppi.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.0))
            elif g.atoms[ia].Z > 10: # Some other atom
                paux.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.2))
        PCpi = PatchCollection(ppi, cmap=cmap, alpha=1., lw=1.2, edgecolor='0.6', **keywords)
        PCpi.set_array(np.zeros(len(ppi)))
        PCpi.set_clim(-cmax, cmax) # colorbar limits
        PCaux = PatchCollection(paux, cmap=cmap, alpha=1., lw=1.2, edgecolor='0.6', **keywords)
        PCaux.set_array(np.zeros(len(paux)))
        PCaux.set_clim(-cmax, cmax) # colorbar limits
        return PCpi, PCaux

    def plot_settings(self, ppi, paux):
        fig = plt.figure(figsize=(8, 6))
        axes = plt.axes()
        x = self.geom.xyz[:, 0]
        y = self.geom.xyz[:, 1]
        bdx = 2
        axes.set_xlim(min(x)-bdx, max(x)+bdx)
        axes.set_ylim(min(y)-bdx, max(y)+bdx)
        plt.rc('font', family='Bitstream Vera Serif', size=16)
        plt.rc('text', usetex=True)
        axes.add_collection(paux)
        axes.add_collection(ppi)
        axes.set_xlabel(r'$x$ (\AA)')
        axes.set_ylabel(r'$y$ (\AA)')
        axes.set_aspect('equal')
        return fig, axes, x, y, bdx


    def plot_polarization(self, cmax=0.4, title=None):
        data = self.nup-self.ndn
        # Patches
        ppi, paux = self.get_atomic_patches(cmax=cmax)
        fig, axes, x, y, bdx = self.plot_settings(ppi, paux)
        ppi.set_array(data) # Set data
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(ppi, label=r'$Q_\uparrow -Q_\downarrow$ ($e$)', cax=cax)
        plt.subplots_adjust(right=0.8)
        if title:
            axes.set_title(title)
        outfn = self.get_label()+'-pol.pdf'
        fig.savefig(outfn)
        print('Wrote', outfn)
        plt.close('all')

    def plot_charge(self, title=None, chgdiff=False):
        # Patches
        ppi, paux = self.get_atomic_patches()
        fig, axes, x, y, bdx = self.plot_settings(ppi, paux)
        # Compute data
        data = self.nup+self.ndn
        if chgdiff:
            # Plot electron charge difference with respect to neutral atom
            for ia in self.pi_geom:
                data[ia] -= self.pi_geom.atoms[ia].Z-5
            outfn = self.get_label()+'-chgdiff.pdf'
            cmax = max(abs(data))
            ppi.set_clim(-cmax, cmax) # colorbar limits
        else:
            # Plot directly electron charge
            outfn = self.get_label()+'-chg.pdf'
            ppi.set_clim(min(data), max(data)) # colorbar limits
        ppi.set_array(data) # Set data
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(ppi, label=r'$Q_\uparrow +Q_\downarrow$ ($e$)', cax=cax)
        plt.subplots_adjust(right=0.8)
        if title:
            axes.set_title(title)
        fig.savefig(outfn)
        print('Wrote', outfn)
        plt.close('all')

    def plot_rs_polarization(self, vz=0, z=1.1, vmax=0.006, grid_unit=0.075, title=None):
        pol = self.nup-self.ndn
        # Patches
        ppi, paux = self.get_atomic_patches(cmap='Greys', facecolor='None')
        fig, axes, x, y, bdx = self.plot_settings(ppi, paux)
        ppi.set_array(pol) # Set data
        # Create pz orbital for each C atom
        r = np.linspace(0, 1.6, 700)
        func = 5 * np.exp(-r * 5)
        orb = sisl.SphericalOrbital(1, (r, func))
        C = sisl.Atom(6, orb)
        B = sisl.Atom(5, orb)
        N = sisl.Atom(7, orb)
        # Change sc cell for plotting purpose
        vx = np.abs((min(x)-bdx) - (max(x)+bdx))
        vy = np.abs((min(y)-bdx) - (max(y)+bdx))
        if vz == 0:
            vz = vx
        geom = self.pi_geom.move([-(min(x)-bdx), -(min(y)-bdx), -self.geom.center()[2]])
        geom.xyz[np.where(np.abs(geom.xyz[:, 2]) < 1e-3), 2] = 0 # z~0 -> z=0
        H = sisl.Hamiltonian(geom)
        H.geom.set_sc(sisl.SuperCell([vx, vy, vz]))
        for ia in H.geom:
            if H.geom.atoms.Z[ia] == 6:
                H.geom.atom.replace(H.geom.atom[ia], C)
            elif H.geom.atoms.Z[ia] == 5:
                H.geom.atom.replace(H.geom.atom[ia], B)
            elif H.geom.atoms.Z[ia] == 7:
                H.geom.atom.replace(H.geom.atom[ia], N)
        grid = sisl.Grid(grid_unit, sc=H.geom.sc)
        vecs = np.zeros((self.sites, self.sites))
        vecs[0, :] = pol
        es = sisl.EigenstateElectron(vecs, np.zeros(self.sites), H)
        es.sub(0).psi(grid)
        index = grid.index([0, 0, z])
        # Plot only the real part of the WF
        ax = axes.imshow(grid.grid[:, :, index[2]].T.real, cmap='seismic', origin='lower', vmax=vmax, vmin=-vmax,
                         extent=[min(x)-bdx, max(x)+bdx, min(y)-bdx, max(y)+bdx])
        plt.colorbar(ax)
        if title:
            axes.set_title(title)
        outfn = self.get_label()+'-rs-pol.pdf'
        fig.savefig(outfn)
        #grid.write('wavefunction.cube') # write to Cube file
        print('Wrote', outfn)
        plt.close('all')

    def real_space_wf(self, ev, vecs, state, label, title, vz=0, z=1.1, vmax=0.006, grid_unit=0.075):
	    # Patches
        ppi, paux = self.get_atomic_patches(cmap='Greys', facecolor='None')
        fig, axes, x, y, bdx = self.plot_settings(ppi, paux)
        # Create pz orbital for each C atom
        r = np.linspace(0, 1.6, 700)
        func = 5 * np.exp(-r * 5)
        orb = sisl.SphericalOrbital(1, (r, func))
        C = sisl.Atom(6, orb)
        B = sisl.Atom(5, orb)
        N = sisl.Atom(7, orb)
        # Change sc cell for plotting purpose
        vx = np.abs((min(x)-bdx) - (max(x)+bdx))
        vy = np.abs((min(y)-bdx) - (max(y)+bdx))
        if vz == 0:
            vz = vx
        geom = self.pi_geom.move([-(min(x)-bdx), -(min(y)-bdx), -self.geom.center()[2]])
        geom.xyz[np.where(np.abs(geom.xyz[:, 2]) < 5e-2), 2] = 0 # z~0 -> z=0
        H = sisl.Hamiltonian(geom)
        H.geom.set_sc(sisl.SuperCell([vx, vy, vz]))
        for ia in H.geom:
            if H.geom.atoms.Z[ia] == 6:
                H.geom.atom.replace(H.geom.atom[ia], C)
            elif H.geom.atoms.Z[ia] == 5:
                H.geom.atom.replace(H.geom.atom[ia], B)
            elif H.geom.atoms.Z[ia] == 7:
                H.geom.atom.replace(H.geom.atom[ia], N)
        grid = sisl.Grid(grid_unit, dtype=np.complex128, sc=H.geom.sc)
        es = sisl.EigenstateElectron(vecs.T, ev, H)
        es.sub(state).psi(grid) # plot the ith wavefunction on the grid.
        index = grid.index([0, 0, z])
         # Plot only the real part of the WF
        custom_map = cmap = mcolors.LinearSegmentedColormap.from_list(name='custom_map', colors =['g', 'white', 'red'], N=100)
        ax = axes.imshow(grid.grid[:, :, index[2]].T.real,
                         cmap=custom_map, origin='lower', vmax=vmax, vmin=-vmax, extent=[min(x)-bdx, max(x)+bdx, min(y)-bdx, max(y)+bdx])
        plt.colorbar(ax)
        axes.set_title(title)
        outfn = self.get_label()+'-rs-wf%s.pdf'%(label)
        fig.savefig(outfn)
        #grid.write('wavefunction.cube') # write to Cube file
        print('Wrote', outfn)
        plt.close('all')

    def plot_rs_wf(self, k=[0, 0, 0], vz=0, z=1.1, vmax=0.006, EnWindow=2.0, f=0.25, grid_unit=0.075, ispin=0, density=True):
        if ispin == 0:
            spin_label = '-up'
            ev, vec = self.Hup.eigh(k=k, eigvals_only=False)
        else:
            spin_label = '-dn'
            ev, vec = self.Hdn.eigh(k=k, eigvals_only=False)
        ev -= self.midgap
        if density:
            label = '-dens'
            vec = np.e**(np.angle(vec)*1j)*np.abs(vec)**2
        else:
            label = ''
            vec =  np.e**(np.angle(vec)*1j)*np.abs(vec)
        # Find states over an energy window
        states = np.where(np.abs(ev) < EnWindow)[0]
        for state in states:
            # Plot both [up,down] states
            title = r'$E-E_\mathrm{mid}=%.4f$ eV, $k=[%.1f,%.1f,%.1f] \pi/a$'%(ev[state], k[0], k[1], k[2])
            self.real_space_wf(ev, vec, state, label+spin_label+'-state%i'%state, title, vz=vz, z=z, vmax=vmax, grid_unit=grid_unit)

    def wf(self, data, title, label, f=3000):
	    # Patches
        ppi, paux = self.get_atomic_patches()
        fig, axes, x, y, bdx = self.plot_settings(ppi, paux)
        if max(data) < -min(data):
            data = -data # change sign of wf to have largest element as positive
        scatter1 = axes.scatter(x, y, data*f, 'r') # pos. part, marker AREA is proportional to data
        scatter2 = axes.scatter(x, y, -data*f, 'g') # neg. part
        axes.set_title(title)
        fnout= self.get_label()+'-wf%s.pdf'%label
        fig.savefig(fnout)
        print('Wrote', fnout)
        plt.close('all')

    def plot_wf(self, k=[0, 0, 0], EnWindow=2.0, f=3000, density=True, ispin=0):
        if not density:
            # We plot directly the wavefunction instead
            f = 1000
        if ispin == 0:
            spin_label = '-up'
            ev, vec = self.Hup.eigh(k=k, eigvals_only=False)
        else:
            spin_label = '-dn'
            ev, vec = self.Hdn.eigh(k=k, eigvals_only=False)
        ev -= self.midgap
        if density:
            label = '-dens'
            vec = np.e**(np.angle(vec)*1j)*np.abs(vec)**2
        else:
            label = ''
            vec = vec = np.e**(np.angle(vec)*1j)*np.abs(vec)
        states = np.where(np.abs(ev) < EnWindow)[0]
        data = np.zeros(len(self.geom))
        Clist = [ia for ia in self.geom if self.geom.atoms[ia].Z in [5, 6, 7]]
        for state in states:
            data[Clist] = vec[:, state].real
            title = '$E-E_\mathrm{mid}=%.4f$ eV, $k=[%.2f,%.2f,%.2f] \pi/a$'%(ev[state], k[0], k[1], k[2])
            self.wf(data, title, label+spin_label+'-state%i'%state, f=f)

    def init_nc(self, fn):
        try:
            self.ncf = NC.Dataset(fn, 'a')
            print('Appending to', fn)
        except:
            print('Initiating', fn)
            ncf = NC.Dataset(fn, 'w')
            ncf.createDimension('unl', None)
            ncf.createDimension('spin', 2)
            ncf.createDimension('sites', len(self.pi_geom))
            ncf.createVariable('hash', 'i8', ('unl',))
            ncf.createVariable('U', 'f8', ('unl',))
            ncf.createVariable('Nup', 'i4', ('unl',))
            ncf.createVariable('Ndn', 'i4', ('unl',))
            ncf.createVariable('Density', 'f8', ('unl', 'spin', 'sites'))
            ncf.createVariable('Etot', 'f8', ('unl',))
            self.ncf = ncf
            ncf.sync()

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

    def save(self):
        myhash, s = self.gethash()
        i = np.where(self.ncf['hash'][:] == myhash)[0]
        if len(i) == 0:
            i = len(self.ncf['hash'][:])
        else:
            i = i[0]
        self.ncf['hash'][i] = myhash
        self.ncf['U'][i] = self.U
        self.ncf['Nup'][i] = self.Nup
        self.ncf['Ndn'][i] = self.Ndn
        self.ncf['Density'][i, 0] = self.nup
        self.ncf['Density'][i, 1] = self.ndn
        self.ncf['Etot'][i] = self.Etot
        self.ncf.sync()
        print('Wrote (U,Nup,Ndn)=(%.2f,%i,%i) data to'%(self.U, self.Nup, self.Ndn), self.fn)

    def read(self):
        myhash, s = self.gethash()
        i = np.where(self.ncf['hash'][:] == myhash)[0]
        if len(i) == 0:
            print('Hash not found:')
            print('...', s)
            self.random_density()
        else:
            print('Found:')
            print('...', s, 'in file')
            i = i[0]
            self.U = self.ncf['U'][i]
            self.nup = self.ncf['Density'][i][0]
            self.ndn = self.ncf['Density'][i][1]
            self.Etot = self.ncf['Etot'][i]
        self.update_spin_hamiltonian()

    def get_1D_band_structure(self, nk=51, data2file=False, projection=None):
        if not projection:
            projection = range(len(self.pi_geom))
        klist = np.linspace(0, 0.5, nk)
        eigs_up = np.empty([len(klist), self.H0.no])
        eigs_dn = np.empty([len(klist), self.H0.no])
        pdos_up = np.empty([len(klist), self.H0.no]) # Site-projected DOS
        pdos_dn = np.empty([len(klist), self.H0.no])
        # Loop over k
        for ik, k in enumerate(klist):
            eigs_up[ik, :], evec = self.Hup.eigh([k, 0, 0], eigvals_only=False)
            v = evec[projection]
            pdos_up[ik, :] = np.diagonal(np.dot(np.conjugate(v).T, v).real)
            eigs_dn[ik, :], evec = self.Hdn.eigh([k, 0, 0], eigvals_only=False)
            v = evec[projection]
            pdos_dn[ik, :] = np.diagonal(np.dot(np.conjugate(v).T, v).real)
        # Set energy reference to midgap
        eigs_up -= self.midgap
        eigs_dn -= self.midgap
        # Save to file
        if data2file:
            fup = open(self.get_label()+'-bands-up.dat', 'w')
            fdn = open(self.get_label()+'-bands-dn.dat', 'w')
            for ik, i in enumerate(klist):
                # write to file
                fup.write('%.8f '%k)
                fdn.write('%.8f '%k)
                for ev in eigs_up[ik]:
                    fup.write('%.8f ' %ev)
                for ev in eigs_dn[ik]:
                    fdn.write('%.8f ' %ev)
                fup.write('\n')
                fdn.write('\n')
            fup.close()
            fdn.close()
        return klist, eigs_up, eigs_dn, pdos_up, pdos_dn

    def plot_bands(self, TSHS=None, nk=51, ymax=4., projection=None):
        fig = plt.figure(figsize=(4, 8))
        axes = plt.axes()
        # Get TB bands
        ka, evup, evdn, pdosup, pdosdn = self.get_1D_band_structure(projection=projection)
        ka = 2*ka # Units ka/pi
        # Add siesta bandstructure?
        if TSHS:
            dftH = sisl.get_sile(TSHS).read_hamiltonian()
            klist = np.linspace(0, 0.5, nk)
            eigs_up = np.empty([len(klist), dftH.no])
            #eigs_dn = np.empty([len(klist), dftH.no])
            print('Diagonalizing DFT Hamiltonian')
            for ik, k in enumerate(klist):
                print('%i/%i'%(ik, nk),)
                eigs_up[ik, :] = dftH.eigh([k, 0, 0], eigvals_only=True)
                #eigs_dn[ik, :] = dftH.eigh([k, 0, 0], eigvals_only=True)
            plt.plot(ka, eigs_up, 'k') # NB: with respect to SIESTA Fermi energy
        if not np.allclose(evup, evdn):
            # Add spin-down component to plot
            plt.plot(ka, evdn, 'g.')
        # Fat bands?
        if projection != None:
            for i, ev in enumerate(evup[0]):
                #plt.fill_between(ka, evup[:, i]+pdosup[:, i], evup[:, i]-pdosup[:, i], alpha=.4, edgecolor='None', facecolor='Grey')
                plt.errorbar(ka, evup[:, i], yerr=pdosup[:, i], alpha=.4, color='Grey')
        # Add spin-up bands
        plt.plot(ka, evup, 'r')
        plt.ylim(-ymax, ymax)
        plt.rc('font', family='Bitstream Vera Serif', size=19)
        plt.rc('text', usetex=True)
        if title == 'default':
            axes.set_title(r'%s $U=%.2f$ eV'%(self.model, self.U), size=19)
        else:
            axes.set_title(title)
        axes.set_xlabel(r'$ka/\pi$')
        axes.set_ylabel(r'$E_{nk}-E_\mathrm{mid}$ (eV)')
        plt.subplots_adjust(left=0.2, top=.95, bottom=0.1, right=0.95)
        outfn = self.get_label()+'-bands.pdf'
        fig.savefig(outfn)
        print('Wrote', outfn)
        plt.close('all')

    def calc_orbital_charge_overlaps(self, k=[0, 0, 0], ispin=0):
        if ispin == 0:
            ev, evec = self.Hup.eigh(k=k, eigvals_only=False)
        else:
            ev, evec = self.Hdn.eigh(k=k, eigvals_only=False)
        # Compute orbital charge overlaps
        L = np.einsum('ia,ia,ib,ib->ab', evec, evec, evec, evec).real
        return ev, L

    def plot_spectrum(self, k=[0, 0, 0], xmax=10., ymax=0.15, annotate=True, title='default'):
        fig = plt.figure(figsize=(10, 5))
        axes = plt.axes()
        axes.fill_between([-xmax, 0], 0, 1.0, facecolor='k', alpha=0.1)
        # Plot data
        lmax = 0.0
        for i in range(2):
            ev, L = self.calc_orbital_charge_overlaps(k, ispin=i)
            ev -= self.midgap
            L = np.diagonal(L)
            lmax = max(lmax, max(L))
            plt.plot(ev, L, 'rg'[i]+'.+'[i], label=[r'$\sigma=\uparrow$', r'$\sigma=\downarrow$'][i])
            if annotate:
                for i in range(len(ev)):
                    axes.annotate(i, (ev[i], L[i]), fontsize=6)
        axes.set_xlabel(r'$E_{\alpha\sigma}-E_\mathrm{mid}$ (eV)')
        axes.set_ylabel(r'$\eta_{\alpha\sigma}=\int dr |\psi_{\alpha\sigma}|^4$')
        axes.legend()
        axes.set_xlim(-xmax, xmax)
        if ymax > 0:
            axes.set_ylim(0, ymax)
        else:
            # Autoset ylim to 110% of lmax
            axes.set_ylim(0, 1.1*lmax)
        plt.rc('font', family='Bitstream Vera Serif', size=19)
        plt.rc('text', usetex=True)
        if title == 'default':
            axes.set_title(r'%s $U=%.2f$ eV'%(self.model, self.U))
        else:
            axes.set_title(title)
        outfn = self.get_label()+'-spectrum.pdf'
        fig.savefig(outfn)
        print('Wrote', outfn)
        plt.close('all')
