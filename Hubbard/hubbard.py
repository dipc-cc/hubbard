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

    def __init__(self, fn, t1=2.7, t2=0.2, t3=0.18, nsc=[1, 1, 1], kmesh=[1, 1, 1], what=None):
        # Save parameters
        if fn[-3:] == '.XV':
            self.fn = fn[:-3]
        elif fn[-4:] == '.xyz':
            self.fn = fn[:-4]
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
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
        self.geom.write('molecule.xyz')
        print('Wrote molecule.xyz')
        # Determine pz sites
        aux = []
        for ia in self.geom:
            if self.geom.atoms[ia].Z not in [5, 6, 7]:
                aux.append(ia)
        # Remove all sites not carbon-type
        self.pi_geom = self.geom.remove(aux)
        self.sites = len(self.pi_geom)
        print('Found %i pz sites' %self.sites)
        # Count number of pi-electrons:
        nB = len(np.where(self.pi_geom.atoms.Z == 5)[0])
        nC = len(np.where(self.pi_geom.atoms.Z == 6)[0])
        nN = len(np.where(self.pi_geom.atoms.Z == 7)[0])
        ntot = 0*nB+1*nC+2*nN
        print('Found %i B-atoms, %i C-atoms, %i N-atoms' %(nB, nC, nN))
        print('Neutral system corresponds to a total of %i electrons' %ntot)
        # Set default values
        self.U = 0.0
        self.Ndn = int(ntot/2)
        self.Nup = int(ntot-self.Ndn)
        print('   U   =', self.U)
        print('   Nup =', self.Nup)
        print('   Ndn =', self.Ndn)
        # Construct Hamiltonians
        self.set_hoppings()
        # Generate kmesh
        [nx, ny, nz] = kmesh
        self.kmesh = []
        for kx in np.arange(0, 1, 1./nx):
            for ky in np.arange(0, 1, 1./ny):
                for kz in np.arange(0, 1, 1./nz):
                    self.kmesh.append([kx, ky, kz])
        # Initialize data file
        self.init_nc(self.fn+'.nc')
        # Try reading from file or use random density
        self.read()
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
        s += '-U%i'%(self.U*100)
        return s

    def polarize(self, pol):
        print('Polarizing by', pol)
        self.Nup += int(pol)
        self.Ndn -= int(pol)
        return self.Nup, self.Ndn

    def set_hoppings(self):
        # Radii defining 1st, 2nd, and 3rd neighbors
        R = [0.1, 1.6, 2.6, 3.1]
        # Build hamiltonian for backbone
        g = self.pi_geom
        self.H0 = sisl.Hamiltonian(g)
        for ia in g:
            idx = g.close(ia, R=R)
            if g.atoms[ia].Z == 5:
                # set onsite for B sites
                self.H0.H[ia, ia] = 3.
                print('Found B site')
            # set onsite for N sites
            if g.atoms[ia].Z == 7:
                self.H0.H[ia, ia] = -3.
                print('Found N site')
            # set hoppings
            self.H0.H[ia, idx[1]] = -self.t1
            if self.t2 != 0:
                self.H0.H[ia, idx[2]] = -self.t2
            if self.t3 != 0:
                self.H0.H[ia, idx[3]] = -self.t3
        self.Hup = self.H0.copy()
        self.Hdn = self.H0.copy()

    def random_density(self):
        self.nup = np.random.rand(self.sites)
        self.nup = self.nup/np.sum(self.nup)*(self.Nup)
        self.ndn = np.random.rand(self.sites)
        self.ndn = self.ndn/np.sum(self.ndn)*(self.Ndn)

    def iterate(self, mix=1.0):
        nup = self.nup
        ndn = self.ndn
        Nup = self.Nup
        Ndn = self.Ndn
        # Update Hamiltonian
        for ia in self.pi_geom:
            # charge on neutral atom:
            n0 = self.pi_geom.atoms[ia].Z-5
            self.Hup.H[ia, ia] = self.H0.H[ia, ia] + self.U*(self.ndn[ia]-n0)
            self.Hdn.H[ia, ia] = self.H0.H[ia, ia] + self.U*(self.nup[ia]-n0)
        # Solve eigenvalue problems
        niup = 0*nup
        nidn = 0*ndn
        for k in self.kmesh:
            ev_up, evec_up = self.Hup.eigh(k=k, eigvals_only=False)
            ev_dn, evec_dn = self.Hdn.eigh(k=k, eigvals_only=False)
            # Compute new occupations
            niup += np.sum(np.absolute(evec_up[:, :int(Nup)])**2, axis=1).real
            nidn += np.sum(np.absolute(evec_dn[:, :int(Ndn)])**2, axis=1).real
        niup = niup/len(self.kmesh)
        nidn = nidn/len(self.kmesh)
        # Measure of density change
        dn = np.sum(abs(nup-niup))
        # Update occupations
        self.nup = mix*niup+(1.-mix)*nup
        self.ndn = mix*nidn+(1.-mix)*ndn
        # Compute total energy
        self.Etot = np.sum(ev_up[:int(Nup)])+np.sum(ev_dn[:int(Ndn)])-self.U*np.sum(nup*ndn)
        return dn, self.Etot

    def get_atomic_patch(self):
        pH = []
        pC = []
        pS = []
        g = self.geom
        for ia in g:
            if g.atoms[ia].Z == 1:
                pH.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.4))
            elif g.atoms[ia].Z == 5: # B
                pC.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.0))
            elif g.atoms[ia].Z == 6: # C
                pC.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.7))
            elif g.atoms[ia].Z == 7: # N
                pC.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=1.0))
            elif g.atoms[ia].Z > 10:
                # Substrate atom
                pS.append(patches.Circle((g.xyz[ia, 0], g.xyz[ia, 1]), radius=0.2))
        return pH, pC, pS

    def plot_polarization(self, f=100, cmax=0.4):
        pH, pC, pS = self.get_atomic_patch()
        pol = self.nup-self.ndn
        fig = plt.figure(figsize=(8, 6))
        axes = plt.axes()
        x = self.geom.xyz[:, 0]
        y = self.geom.xyz[:, 1]
        bdx = 2
        axes.set_xlim(min(x)-bdx, max(x)+bdx)
        axes.set_ylim(min(y)-bdx, max(y)+bdx)
        plt.rc('font', family='Bitstream Vera Serif', size=16)
        plt.rc('text', usetex=True)
        pc1 = PatchCollection(pH, cmap=plt.cm.bwr, alpha=1., lw=1.2, edgecolor='0.6')
        pc1.set_array(np.zeros(len(pH)))
        axes.add_collection(pc1)
        pc2 = PatchCollection(pC, cmap=plt.cm.bwr, alpha=1., lw=1.2, edgecolor='0.6')
        pc2.set_array(pol)
        pc1.set_clim(-cmax, cmax) # colorbar limits
        pc2.set_clim(-cmax, cmax) # colorbar limits
        axes.add_collection(pc2)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(pc2, label=r'$Q_\uparrow -Q_\downarrow$ ($e$)', cax=cax)
        plt.subplots_adjust(right=0.8)
        axes.set_xlabel(r'$x$ (\AA)')
        axes.set_ylabel(r'$y$ (\AA)')
        axes.set_aspect('equal')
        outfn = self.get_label()+'-pol.pdf'
        fig.savefig(outfn)
        print('Wrote', outfn)
        plt.close('all')

    def plot_rs_polarization(self, vz=0, z=1.1, vmax=0.006, grid_unit=0.075):
        pH, pC, pS = self.get_atomic_patch()
        pol = self.nup-self.ndn
        fig = plt.figure(figsize=(8, 6))
        axes = plt.axes()
        x = self.geom.xyz[:, 0]
        y = self.geom.xyz[:, 1]
        bdx = 2
        plt.rc('font', family='Bitstream Vera Serif', size=16)
        plt.rc('text', usetex=True)
        # Plot geometry backbone
        pc1 = PatchCollection(pH, cmap='Greys', alpha=1., lw=1.2, facecolor='None')
        pc1.set_array(np.zeros(len(pH)))
        axes.add_collection(pc1)
        pc2 = PatchCollection(pC, cmap='Greys', alpha=1., lw=1.2, facecolor='None')
        pc2.set_array(np.zeros(len(pC)))
        axes.add_collection(pc2)
        pc1.set_clim(-10, 10) # colorbar limits
        pc2.set_clim(-10, 10) # colorbar limits
        # Create pz orbital for each C atom
        r = np.linspace(0, 1.6, 700)
        func = 5 * np.exp(-r * 5)
        orb = sisl.SphericalOrbital(1, (r, func))
        C = sisl.Atom(6, orb)
        # Change sc cell for plotting purpose
        vx = np.abs((min(x)-bdx) - (max(x)+bdx))
        vy = np.abs((min(y)-bdx) - (max(y)+bdx))
        if vz == 0:
            vz = vx
        geom = self.pi_geom.move([-(min(x)-bdx), -(min(y)-bdx), -self.geom.center()[2]])
        geom.xyz[np.where(np.abs(geom.xyz[:, 2]) < 1e-3), 2] = 0 # z~0 -> z=0
        H = sisl.Hamiltonian(geom)
        H.geom.set_sc(sisl.SuperCell([vx, vy, vz]))
        H.geom.atom.replace(H.geom.atom[0], C)
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
        axes.set_xlim(min(x)-bdx, max(x)+bdx)
        axes.set_ylim(min(y)-bdx, max(y)+bdx)
        axes.set_xlabel(r'$x$ (\AA)')
        axes.set_ylabel(r'$y$ (\AA)')
        outfn = self.get_label()+'-rs-pol.pdf'
        fig.savefig(outfn)
        #grid.write('wavefunction.cube') # write to Cube file
        print('Wrote', outfn)
        plt.close('all')

    def real_space_wf(self, ev, vecs, state, spinLabel, title, vz=0, z=1.1, vmax=0.006, grid_unit=0.075):
        pH, pC, pS = self.get_atomic_patch()
        fig = plt.figure(figsize=(8, 6))
        axes = plt.axes()
        x = self.geom.xyz[:, 0]
        y = self.geom.xyz[:, 1]
        bdx = 2
        plt.rc('font', family='Bitstream Vera Serif', size=16)
        plt.rc('text', usetex=True)
        # Plot geometry backbone
        pc1 = PatchCollection(pH, cmap='Greys', alpha=1., lw=1.2, facecolor='None')
        pc1.set_array(np.zeros(len(pH)))
        axes.add_collection(pc1)
        pc2 = PatchCollection(pC, cmap='Greys', alpha=1., lw=1.2, facecolor='None')
        pc2.set_array(np.zeros(len(pC)))
        axes.add_collection(pc2)
        pc1.set_clim(-10, 10) # colorbar limits
        pc2.set_clim(-10, 10) # colorbar limits
        # Create pz orbital for each C atom
        r = np.linspace(0, 1.6, 700)
        func = 5 * np.exp(-r * 5)
        orb = sisl.SphericalOrbital(1, (r, func))
        C = sisl.Atom(6, orb)
        # Change sc cell for plotting purpose
        vx = np.abs((min(x)-bdx) - (max(x)+bdx))
        vy = np.abs((min(y)-bdx) - (max(y)+bdx))
        if vz == 0:
            vz = vx
        geom = self.pi_geom.move([-(min(x)-bdx), -(min(y)-bdx), -self.geom.center()[2]])
        geom.xyz[np.where(np.abs(geom.xyz[:, 2]) < 1e-3), 2] = 0 # z~0 -> z=0
        H = sisl.Hamiltonian(geom)
        H.geom.set_sc(sisl.SuperCell([vx, vy, vz]))
        H.geom.atom.replace(H.geom.atom[0], C)
        if np.any(vecs.imag):
            dtype=np.complex128
        else:
            dtype=None
        grid = sisl.Grid(grid_unit, dtype=dtype, sc=H.geom.sc)
        es = sisl.EigenstateElectron(vecs.T, ev, H)
        es.sub(state).psi(grid) # plot the ith wavefunction on the grid.
        index = grid.index([0, 0, z])
         # Plot only the real part of the WF
        ax = axes.imshow(grid.grid[:, :, index[2]].T.real,
                         cmap='seismic', origin='lower', vmax=vmax, vmin=-vmax, extent=[min(x)-bdx, max(x)+bdx, min(y)-bdx, max(y)+bdx])
        plt.colorbar(ax)
        axes.set_title(title)
        axes.set_xlim(min(x)-bdx, max(x)+bdx)
        axes.set_ylim(min(y)-bdx, max(y)+bdx)
        axes.set_xlabel(r'$x$ (\AA)')
        axes.set_ylabel(r'$y$ (\AA)')
        outfn = self.get_label()+'-rs-wf-%s-state%i.pdf'%(spinLabel, state)
        fig.savefig(outfn)
        #grid.write('wavefunction.cube') # write to Cube file
        print('Wrote', outfn)
        plt.close('all')

    def plot_rs_wf(self, k=[0, 0, 0], vz=0, z=1.1, vmax=0.006, EnWindow=2.0, f=0.25, grid_unit=0.075):
        egap, emid = self.find_midgap()
        evup, vecup = self.Hup.eigh(k=k, eigvals_only=False)
        evdn, vecdn = self.Hdn.eigh(k=k, eigvals_only=False)
        evup -= emid
        evdn -= emid
        # Find states over an energy window
        states = np.where(np.abs(evup) < EnWindow)[0]
        for state in states:
            # Plot both [up,down] states
            title = r'$E-E_{mid}=%.4f$ eV, $k=[%.1f,%.1f,%.1f] \pi/a$'%(evup[state], k[0], k[1], k[2])
            self.real_space_wf(evup, vecup, state, 'up', title, vz=vz, z=z, vmax=vmax, grid_unit=grid_unit)
            title = r'$E-E_{mid}=%.4f$ eV, $k=[%.1f,%.1f,%.1f] \pi/a$'%(evdn[state], k[0], k[1], k[2])
            self.real_space_wf(evdn, vecdn, state, 'dn', title, vz=vz, z=z, vmax=vmax, grid_unit=grid_unit)

    def plot_charge(self, f=100):
        pH, pC, pS = self.get_atomic_patch()
        x = self.pi_geom.xyz[:, 0]
        y = self.pi_geom.xyz[:, 1]
        bdx = 2
        chg = self.nup+self.ndn
        fig = plt.figure(figsize=(6, 6))
        axes = plt.axes()
        # Plot geometry backbone
        pc1 = PatchCollection(pH, cmap=plt.cm.bwr, alpha=1., lw=1.2, edgecolor='0.6')
        pc1.set_array(np.zeros(len(pH)))
        axes.add_collection(pc1)
        pc2 = PatchCollection(pC, cmap=plt.cm.bwr, alpha=1., lw=1.2, edgecolor='0.6')
        pc2.set_array(np.zeros(len(pC)))
        axes.add_collection(pc2)
        pc1.set_clim(-10, 10) # colorbar limits
        pc2.set_clim(-10, 10) # colorbar limits
        axes.set_aspect('equal')
        plt.rc('font', family='Bitstream Vera Serif', size=16)
        plt.rc('text', usetex=True)
        scatter = axes.scatter(x, y, f*chg, 'r') # pos. part, marker AREA is proportional to data
        axes.set_xlim(min(x)-bdx, max(x)+bdx)
        axes.set_ylim(min(y)-bdx, max(y)+bdx)
        axes.set_xlabel(r'$x$ (\AA)')
        axes.set_ylabel(r'$y$ (\AA)')
        outfn =	self.get_label()+'-chg.pdf'
        fig.savefig(outfn)
        print('Wrote', outfn)
        plt.close('all')

    def wf(self, data, title, label, f=3000):
        pH, pC, pS = self.get_atomic_patch()
        bdx = 2
        x = self.geom.xyz[:, 0]
        y = self.geom.xyz[:, 1]
        fig = plt.figure(figsize=(8, 6))
        axes = plt.axes()
        plt.rc('font', family='Bitstream Vera Serif', size=16)
        plt.rc('text', usetex=True)
        axes.set_xlabel(r'$x$ (\AA)')
        axes.set_ylabel(r'$y$ (\AA)')
        axes.set_xlim(min(x)-bdx, max(x)+bdx)
        axes.set_ylim(min(y)-bdx, max(y)+bdx)
        axes.set_aspect('equal')
        pc1 = PatchCollection(pH, cmap=plt.cm.bwr, alpha=1., lw=1.2, edgecolor='0.6')
        pc1.set_array(np.zeros(len(pH)))
        axes.add_collection(pc1)
        pc2 = PatchCollection(pC, cmap=plt.cm.bwr, alpha=1., lw=1.2, edgecolor='0.6')
        pc2.set_array(np.zeros(len(pC)))
        pc1.set_clim(-10, 10) # colorbar limits
        pc2.set_clim(-10, 10) # colorbar limits
        axes.add_collection(pc2)
        if max(data) < -min(data):
            data = -data # change sign of wf to have largest element as positive
        scatter1 = axes.scatter(x, y, data*f, 'r') # pos. part, marker AREA is proportional to data
        scatter2 = axes.scatter(x, y, -data*f, 'g') # neg. part
        axes.set_title(title)
        fnout= self.get_label()+'-wf%s.pdf'%label
        fig.savefig(fnout)
        print('Wrote', fnout)
        plt.close('all')

    def plot_wf(self, k=[0, 0, 0], EnWindow=2.0, f=3000, density=True):
        if not density:
            # We plot directly the wavefunction instead
            f = 1000
        egap, emid = self.find_midgap()
        evup, vecup = self.Hup.eigh(k=k, eigvals_only=False)
        evdn, vecdn = self.Hdn.eigh(k=k, eigvals_only=False)
        evup -= emid
        evdn -= emid
        states = np.where(np.abs(evup) < EnWindow)[0]
        data = np.zeros(len(self.geom))
        Clist = [ia for ia in self.geom if self.geom.atoms[ia].Z in [5, 6, 7]]
        for state in states:
            # Plot both [up,down] states
            if density:
                data[Clist] = np.sign(vecup[:, state].real)*(vecup[:, state].real)**2
            else:
                data[Clist] = vecup[:, state].real
            title = '$E-E_{mid}=%.4f$ eV, $k=[%.2f,%.2f,%.2f] \pi/a$'%(evup[state], k[0], k[1], k[2])
            self.wf(data, title, '-up-state%i'%state, f=f)
            if density:
                data[Clist] = np.sign(vecdn[:, state].real)*(vecdn[:, state].real)**2
            else:
                data[Clist] = vecdn[:, state].real
                title = '$E-E_{mid}=%.4f$ eV, $k=[%.2f,%.2f,%.2f] \pi/a$'%(evdn[state], k[0], k[1], k[2])
            self.wf(data, title, '-dn-state%i'%state, f=f)

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
            i = i[0]
            self.U = self.ncf['U'][i]
            self.nup = self.ncf['Density'][i][0]
            self.ndn = self.ncf['Density'][i][1]
            self.Etot = self.ncf['Etot'][i]

    def find_midgap(self, k=[0, 0, 0], verbose=False):
        evup = self.Hup.eigh(k=k)
        evdn = self.Hdn.eigh(k=k)
        homo = max(evup[self.Nup-1], evdn[self.Ndn-1])
        lumo = min(evup[self.Nup], evdn[self.Ndn])
        if verbose:
            print('HL gap: %.3f eV at k=' % (lumo-homo), k)
        return lumo-homo, (lumo+homo)/2

    def get_1D_band_structure(self, nk=51):
        # Save to file
        fup = open(self.get_label()+'-bands-up.dat', 'w')
        fdn = open(self.get_label()+'-bands-dn.dat', 'w')
        klist = np.linspace(0, 0.5, nk)
        eigs_up = np.empty([len(klist), self.H0.no])
        eigs_dn = np.empty([len(klist), self.H0.no])
        gap1, emid = self.find_midgap()
        gap0 = 1.e10
        k0 = -1
        for ik, k in enumerate(klist):
            eigs_up[ik, :] = self.Hup.eigh([k, 0, 0], eigvals_only=True)
            eigs_dn[ik, :] = self.Hdn.eigh([k, 0, 0], eigvals_only=True)
            egap, tmp = self.find_midgap(k=[k, 0, 0])
            if egap < gap0:
                gap0 = egap
                k0 = k
            fup.write('%.8f '%k)
            fdn.write('%.8f '%k)
            for ev in eigs_up[ik]:
                fup.write('%.8f ' %(ev-emid))
            for ev in eigs_dn[ik]:
                fdn.write('%.8f ' %(ev-emid))
            fup.write('\n')
            fdn.write('\n')
        print('   Gap1: %.3f eV at k=[0,0,0]' % gap1)
        print('   Gap0: %.3f eV at k=[%.3f,0,0]' % (gap0, k0))
        return klist, eigs_up, eigs_dn

    def plot_bands(self, TSHS=None, nk=51):
        fig = plt.figure(figsize=(4, 8))
        axes = plt.axes()
        # Get TB bands
        ka, evup, evdn = self.get_1D_band_structure()
        ka = 2*ka # Units ka/pi
        # determine midgap
        egap, emid = self.find_midgap()
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
            #print
            plt.plot(ka, eigs_up, 'k')
        plt.plot(ka, evup-emid, 'r')
        plt.ylim(-4, 4)
        plt.rc('font', family='Bitstream Vera Serif', size=19)
        plt.rc('text', usetex=True)
        axes.set_title(r'%s $U=%.2f$ eV'%(self.model, self.U), size=19)
        axes.set_xlabel(r'$ka/\pi$')
        axes.set_ylabel(r'$E_{nk}$ (eV)')
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

    def plot_localizations(self, k=[0, 0, 0], ymax=0.15, annotate=True):
        fig = plt.figure(figsize=(10, 5))
        axes = plt.axes()
        axes.fill_between([-10, 0], 0, ymax, facecolor='k', alpha=0.1)
        # Plot data
        egap, emid = self.find_midgap()
        for i in range(2):
            ev, L = self.calc_orbital_charge_overlaps(k, ispin=i)
            L = np.diagonal(L)
            print(i, max(L))
            plt.plot(ev-egap, L, 'rg'[i]+'.+'[i], label=[r'$\sigma=\uparrow$', r'$\sigma=\downarrow$'][i])
            if annotate:
                for i in range(len(ev)):
                    axes.annotate(i, (ev[i]-emid, L[i]), fontsize=6)
        axes.set_xlabel(r'$E_{\alpha\sigma}$ (eV)')
        axes.set_ylabel(r'$\eta_{\alpha\sigma}=\int dr |\psi_{\alpha\sigma}|^4$')
        axes.legend()
        axes.set_xlim(-10, 10)
        axes.set_ylim(0, ymax)
        plt.rc('font', family='Bitstream Vera Serif', size=19)
        plt.rc('text', usetex=True)
        axes.set_title(r'%s $U=%.2f$ eV'%(self.model, self.U))
        outfn = self.get_label()+'-loc.pdf'
        fig.savefig(outfn)
        print('Wrote', outfn)
        plt.close('all')
