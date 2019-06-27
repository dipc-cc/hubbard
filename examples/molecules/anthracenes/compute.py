import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2
import sys
import numpy as np
import Hubbard.ncdf as ncdf
import sisl

# Build sisl Geometry object
mol_file = '2-anthracene'
mol = sisl.get_sile(mol_file+'.XV').read_geometry()
mol.sc.set_nsc([1,1,1])
mol = mol.move(-mol.center(what='xyz'))

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
H = hh.HubbardHamiltonian(Hsp2)

# Output file to collect the energy difference between
# FM and AFM solutions
f = open('FM-AFM.dat', 'w')

for u in np.linspace(0.0, 4.0, 5):
    # We approach the solutions from above, starting at U=4eV
    H.U = 4.0-u

    # AFM case first
    try:
        c = ncdf.read(mol_file+'.nc', ncgroup='AFM_U%i'%(H.U*1000)) # Try reading, if we already have density on file
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()
    dn = H.converge()
    eAFM = H.Etot
    ncdf.write(H, mol_file+'.nc', ncgroup='AFM_U%i'%(H.U*1000))
    
    # Now FM case
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1
    try:
        c = ncdf.read(mol_file+'.nc', ncgroup='FM_U%i'%(H.U*1000)) # Try reading, if we already have density on file
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()
    dn = H.converge()
    eFM = H.Etot
    ncdf.write(H, mol_file+'.nc', ncgroup='FM_U%i'%(H.U*1000))
    
    # Revert the imbalance for next loop
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f\n'%(4.0-u, eFM-eAFM))

f.close()
