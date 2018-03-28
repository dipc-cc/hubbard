import numpy as np
import netCDF4 as NC
import matplotlib.pyplot as plt
#import glob
import sisl


class Hubbard(object):
    
    def __init__(self, fn, t=[0,2.7,0.2,0.18], R=[0.1,1.6,2.6,3.1], U=5., nsc=[1,1,1]):
        # Save parameters
        self.fn = fn
        self.t = t # [onsite,1NN,2NN,3NN]
        self.R = R # [0.1,r1,r2,r3]
        self.U = U # Onsite Coulomb repulsion
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
        # Set hoppings
        self.set_hoppings()
        self.conv = {}
        
    def set_hoppings(self):
        # Build hamiltonian for backbone
        self.H0 = sisl.Hamiltonian(self.pi_geom)
        for ia in self.pi_geom:
            idx = self.pi_geom.close(ia,R=self.R)
            for j,ti in enumerate(self.t):
                if ti != 0:
                    self.H0.H[ia,idx[j]] = -ti

    def init_spins(self,dN):
        self.Hup = self.H0.copy()
        self.Hdn = self.H0.copy()
        # Determine how many up's and down's
        sites = len(self.pi_geom)
        self.Nup = int(sites/2+dN)
        self.Ndn = int(sites-self.Nup)
        print 'Nup, Ndn, dN =',self.Nup, self.Ndn, self.Nup-self.Ndn
        assert self.Nup+self.Ndn == sites
        self.nup = np.random.rand(sites)
        self.nup = self.nup/np.sum(self.nup)*self.Nup
        self.ndn = np.random.rand(sites)
        self.ndp = self.ndn/np.sum(self.ndn)*self.Ndn

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
        ev_up, evec_up = self.Hup.eigh(eigvals_only=False)
        ev_dn, evec_dn = self.Hdn.eigh(eigvals_only=False)
        # Compute new occupations
        niup = np.sum(np.power(evec_up[:,:int(Nup)],2),axis=1).real
        nidn = np.sum(np.power(evec_dn[:,:int(Ndn)],2),axis=1).real
        # Measure of density change
        dn = np.sum(abs(nup-niup))
        # Update occupations
        self.nup = mix*niup+(1.-mix)*nup
        self.ndn = mix*nidn+(1.-mix)*ndn
        # Compute total energy
        self.Etot = np.sum(ev_up[:int(Nup)])+np.sum(ev_dn[:int(Ndn)])-self.U*np.sum(nup*ndn)
        return dn,self.Etot
        
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
        
    def save_state(self):
        self.conv[self.U] = [1*self.nup,1*self.ndn,self.Etot]
        print 'Saved state for U = %.4f eV' %self.U
        
    def retrieve_state(self,U):
        [self.nup,self.dn,Etot] = self.conv[U]
        self.U = U
        print 'Retrieved state for U = %.4f eV' %self.U

    def init_nc(self,fn):
        ncf = NC.Dataset(fn,'w')
        ncf.createDimension('unl',None)
        ncf.createDimension('spin',2)
        ncf.createDimension('sites',len(self.pi_geom))
        ncf.createVariable('U', 'f8', ("unl",))
        ncf.createVariable('density', 'f8', ("unl","spin","sites"))
        self.ncf = ncf
        
    def save(self):
        i = len(self.ncf['U'][:])
        print "entries =", i
        j, = np.where(self.ncf['U'][:] == self.U)
        if len(j) == 0:
            self.ncf['U'][i] = self.U
            self.ncf['density'][i] = [self.nup,self.ndn]
        self.ncf.sync()
        

if __name__ == '__main__':
     T = Hubbard('molecule.XV')
     T.init_spins(1)
     [T.iterate(mix=.1) for i in range(30)]
     T.init_nc('test.nc')
     T.save()
