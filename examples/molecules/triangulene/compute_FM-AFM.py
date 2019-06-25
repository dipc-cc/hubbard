import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.ncdf as ncdf
import sys
import numpy as np
import sisl

# Folder name (triangulene-1, -2 or -3)
fn = sys.argv[1]

# Build sisl Geometry object
mol = sisl.get_sile(fn+'/molecule.xyz').read_geometry()
mol = mol.move(-mol.center(what='xyz'))
mol.sc.set_nsc([1,1,1])

# 3NN tight-binding model
H = hh.HubbardHamiltonian(mol, t1=2.7, t2=.2, t3=.18)

f = open(fn+'/FM-AFM.dat', 'w')

for u in np.linspace(0.0, 3.5, 15):
    H.U = u
    # AFM case first
    try:
        c = ncdf.read(fn+'/triangulene.nc', ncgroup='AFM_U%i'%(u*100)) # Try reading, if we already have density on file
        assert ncdf.gethash(H).hash == c.hash
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()

    dn = H.converge(tol=1e-10)
    eAFM = H.Etot
    ncdf.write(H, fn+'/triangulene.nc', ncgroup='AFM_U%i'%(u*100))

    p = plot.SpinPolarization(H,  colorbar=True)
    p.annotate()
    p.savefig(fn+'/AFM-pol-%i.pdf'%(u*100))

    # Now FM case
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1
    try:
        c = ncdf.read(fn+'/triangulene.nc', ncgroup='FM_U%i'%(u*100)) # Try reading, if we already have density on file
        assert ncdf.gethash(H).hash == c.hash
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()
    dn = H.converge(tol=1e-10)
    eFM = H.Etot
    ncdf.write(H, fn+'/triangulene.nc', ncgroup='FM_U%i'%(u*100))

    # Revert the imbalance for next loop
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()
