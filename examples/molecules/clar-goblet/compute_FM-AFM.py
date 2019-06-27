import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import Hubbard.ncdf as ncdf
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('clar-goblet.xyz').read_geometry()
mol.sc.set_nsc([1,1,1])
mol = mol.move(-mol.center(what='xyz'))

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
H = hh.HubbardHamiltonian(Hsp2)

f = open('FM-AFM.dat', 'w')

for u in np.linspace(0.0, 3.5, 15):
    # We approach the solutions for different U values
    H.U = u
    try:
        c = ncdf.read('clar-goblet.nc', ncgroup='AFM_U%i'%(H.U*1000)) # Try reading, if we already have density on file
        # Use // c.hash == ncdf.gethash(H).hash // To ensure that it is following the same calculation  
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()
    
    # AFM case first
    dn = H.converge(tol=1e-10)
    eAFM = H.Etot
    ncdf.write(H, 'clar-goblet.nc', ncgroup='AFM_U%i'%(H.U*1000))

    # Now FM case
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1
    try:
        c = ncdf.read('clar-goblet.nc', ncgroup='FM_U%i'%(H.U*1000)) # Try reading, if we already have density on file
        # Use // c.hash == ncdf.gethash(H).hash // To ensure that it is following the same calculation 
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()
    dn = H.converge(tol=1e-10)
    eFM = H.Etot
    ncdf.write(H, 'clar-goblet.nc', ncgroup='FM_U%i'%(H.U*1000))

    # Revert the imbalance for next loop
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()
