import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.ncdf as ncdf
import Hubbard.sp2 as sp2
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
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
f = open(fn+'/FM-AFM.dat', 'w')

for u in np.linspace(0.0, 5.0, 21):
    H = hh.HubbardHamiltonian(Hsp2)

    H.U = u
    # AFM case first
    H.read_density(fn+'/triangulene-AFM.nc')
    dn = H.converge(tol=1e-10)
    eAFM = H.Etot
    H.write_density(fn+'/triangulene-AFM.nc')

    p = plot.SpinPolarization(H,  colorbar=True, vmax=0.4)
    p.annotate()
    p.savefig(fn+'/AFM-pol-%i.pdf'%(u*100))

    # Now FM case
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1
    
    H.read_density(fn+'/triangulene-FM.nc')
    dn = H.converge(tol=1e-10)
    eFM = H.Etot
    H.write_density(fn+'/triangulene-FM.nc')

    p = plot.SpinPolarization(H,  colorbar=True, vmax=0.4)
    p.annotate()
    p.savefig(fn+'/FM-pol-%i.pdf'%(u*100))

    # Revert the imbalance for next loop
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()
