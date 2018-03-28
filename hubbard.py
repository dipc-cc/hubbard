import numpy as np
import netCDF4 as NC
import matplotlib.pyplot as plt
#import glob
import sisl
import hashlib

class Hubbard(object):
    
    def __init__(self, fn, t1=2.7, t2=0.2, t3=0.18, nsc=[1,1,1], kmesh=[1,1,1]):
        # Save parameters
        self.fn = fn
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.geom = sisl.get_sile(fn).read_geom()
        self.geom.sc.set_nsc(nsc)
        # Determine pz sites
        Hlist = []
        for ia in self.geom:
            if self.geom.atoms[ia].Z == 1:
                Hlist.append(ia)
        self.pi_geom = self.geom.remove(Hlist)
        self.sites = len(self.pi_geom)
        print 'Found %i pz sites' %self.sites
        # Set default values
        self.U = 0.0
        self.Ndn = int(self.sites/2)
        self.Nup = int(self.sites-self.Ndn)
        print '   U   =', self.U
        print '   Nup =', self.Nup
        print '   Ndn =', self.Ndn
        # Construct Hamiltonians
        self.set_hoppings()
        # Generate kmesh
        [nx,ny,nz] = kmesh
        self.kmesh = []
        for kx in np.arange(0, 1, 1./nx):
            for ky in np.arange(0, 1, 1./ny):
                for kz in np.arange(0, 1, 1./nz):
                    self.kmesh.append([kx, ky, kz])
        # Initialize data file
        self.init_nc(fn+'.nc')
        # First time we need to initialize arrays
        self.random_density()
        # Try reading from file
        self.read()

    def polarize(self,pol):
        'Polarizing by',pol
        self.Nup += int(pol)
        self.Ndn -= int(pol)
        return self.Nup, self.Ndn

    def set_hoppings(self):
        # Radii defining 1st, 2nd, and 3rd neighbors
        R = [0.1,1.6,2.6,3.1]
        # Build hamiltonian for backbone
        self.H0 = sisl.Hamiltonian(self.pi_geom)
        for ia in self.pi_geom:
            idx = self.pi_geom.close(ia,R=R)
            self.H0.H[ia,idx[1]] = -self.t1
            if self.t2 != 0:
                self.H0.H[ia,idx[2]] = -self.t2
            if self.t3 != 0:
                self.H0.H[ia,idx[3]] = -self.t3
        self.Hup = self.H0.copy()
        self.Hdn = self.H0.copy()


    def random_density(self):
        self.nup = np.random.rand(self.sites)
        self.nup = self.nup/np.sum(self.nup)*(self.Nup)
        self.ndn = np.random.rand(self.sites)
        self.ndp = self.ndn/np.sum(self.ndn)*(self.Ndn)

    def iterate(self,mix=1.0):
        nup = self.nup
        ndn = self.ndn
        Nup = self.Nup
        Ndn = self.Ndn
        # Update Hamiltonian
        for ia in self.pi_geom:
            self.Hup.H[ia,ia] = self.U*self.ndn[ia]
            self.Hdn.H[ia,ia] = self.U*self.nup[ia]
        # Solve eigenvalue problems
        niup = 0*nup
        nidn = 0*ndn
        for k in self.kmesh:
            ev_up, evec_up = self.Hup.eigh(k=k,eigvals_only=False)
            ev_dn, evec_dn = self.Hdn.eigh(k=k,eigvals_only=False)
            # Compute new occupations
            niup += np.sum(np.absolute(evec_up[:,:int(Nup)])**2,axis=1).real
            nidn += np.sum(np.absolute(evec_dn[:,:int(Ndn)])**2,axis=1).real
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
        
    def plot_polarization(self,f=100):
        x = self.pi_geom.xyz[:,0]
        y = self.pi_geom.xyz[:,1]
        pol = self.nup-self.ndn
        fig = plt.figure(figsize=(6,6))
        axes = plt.axes()
        axes.set_aspect('equal')
        scatter1 = axes.scatter(x,y,f*pol,'r'); # pos. part, marker AREA is proportional to data
        scatter2 = axes.scatter(x,y,-f*pol,'g'); # neg. part
        plt.show()

    def plot_charge(self,f=100):
        x = self.pi_geom.xyz[:,0]
        y = self.pi_geom.xyz[:,1]
        pol = self.nup+self.ndn
        fig = plt.figure(figsize=(6,6))
        axes = plt.axes()
        axes.set_aspect('equal')
        scatter1 = axes.scatter(x,y,f*pol,'r'); # pos. part, marker AREA is proportional to data
        scatter2 = axes.scatter(x,y,-f*pol,'g'); # neg. part
        plt.show()

    def init_nc(self,fn):
        self.fn = fn
        try:
            self.ncf = NC.Dataset(fn,'a')
            print 'Appending to', fn
        except:
            print 'Initiating', fn
            ncf = NC.Dataset(fn,'w')
            ncf.createDimension('unl',None)
            ncf.createDimension('spin',2)
            ncf.createDimension('sites',len(self.pi_geom))
            ncf.createVariable('hash', 'i8', ('unl',))
            ncf.createVariable('U', 'f8', ('unl',))
            ncf.createVariable('Nup', 'i4', ('unl',))
            ncf.createVariable('Ndn', 'i4', ('unl',))
            ncf.createVariable('Density', 'f8', ('unl','spin','sites'))
            ncf.createVariable('Etot', 'f8', ('unl',))
            self.ncf = ncf
        
    def save(self):
        s = 'U%.4f Nup%i Ndn%i' %(self.U, self.Nup, self.Ndn)
        myhash = int(hashlib.md5(s).hexdigest()[:7],16)
        # Check if this set is already stored
        i = np.where(self.ncf['hash'][:] == myhash)[0]
        if len(i) == 0:
            i = len(self.ncf['hash'][:])
        else:
            i = i[0] # first entry
        self.ncf['hash'][i] = myhash
        self.ncf['U'][i] = self.U
        self.ncf['Nup'][i] = self.Nup
        self.ncf['Ndn'][i] = self.Ndn
        self.ncf['Density'][i,0] = self.nup
        self.ncf['Density'][i,1] = self.ndn
        self.ncf['Etot'][i] = self.Etot
        self.ncf.sync()
        print 'Wrote (U,Nup,Ndn)=(%.2f,%i,%i) data to'%(self.U,self.Nup,self.Ndn), self.fn

    def read(self):
        s = 'U%.4f Nup%i Ndn%i' %(self.U, self.Nup, self.Ndn)
        myhash = int(hashlib.md5(s).hexdigest()[:7],16)
        i = np.where(self.ncf['hash'][:] == myhash)[0]
        if len(i) == 0:
            print 'Data not found'
        else:
            i = i[0]
            self.U = self.ncf['U'][i]
            self.nup = self.ncf['Density'][i][0]
            self.ndn = self.ncf['Density'][i][1]
            self.Etot = self.ncf['Etot'][i]


    def get_1D_band_structure(self):
        klist = np.linspace(0,0.5,101)
        eigs_up = np.empty([len(klist), self.H0.no])
        eigs_dn = np.empty([len(klist), self.H0.no])
        for ik, k in enumerate(klist):
            eigs_up[ik,:] = self.Hup.eigh([k,0,0], eigvals_only=True)
            eigs_dn[ik,:] = self.Hdn.eigh([k,0,0], eigvals_only=True)
        return klist, eigs_up, eigs_dn


if __name__ == '__main__':
    T = Hubbard('molecule.XV')
    print T.polarize(1)
    T.iterate()
    T.save()
    T.U = 3.2
    T.read()
    for i in range(10):
        print T.iterate(mix=1)
    T.save()
    Nup, Ndn = T.polarize(1)
    T.read()
    for i in range(10):
        print T.iterate(mix=.1)
    T.save()
