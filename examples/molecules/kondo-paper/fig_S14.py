import Hubbard.hamiltonian as hh
import Hubbard.ncdf as ncdf
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('junction-2-2.XV').read_geometry()
mol.sc.set_nsc([1,1,1])

# 3NN tight-binding model
H = hh.HubbardHamiltonian(mol, t1=2.7, t2=0.2, t3=.18)

# Output file to collect the energy difference between
# FM and AFM solutions
f = open('FM-AFM.dat', 'w')

for u in np.linspace(0.0, 1.4, 15):
    # We approach the solutions from above, starting at U=4eV
    H.U = 4.4-u
    # AFM case first
    try:
        c = ncdf.read('fig_S14.nc', ncgroup='AFM_U%i'%(H.U*1000)) # Try reading, if we already have density on file
        # Use // c.hash == ncdf.gethash(H).hash // To ensure that it is following the same calculation  
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()

    dn = H.converge()
    eAFM = H.Etot
    ncdf.write(H, 'fig_S14.nc', ncgroup='AFM_U%i'%(H.U*1000))

    # Now FM case
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1
    try:
        c = ncdf.read('fig_S14.nc', ncgroup='FM_U%i'%(H.U*1000)) # Try reading, if we already have density on file
        # Use // c.hash == ncdf.gethash(H).hash // To ensure that it is following the same calculation  
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()

    dn = H.converge()
    eFM = H.Etot
    ncdf.write(H, 'fig_S14.nc', ncgroup='FM_U%i'%(H.U*1000))

    # Revert the imbalance for next loop
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()
