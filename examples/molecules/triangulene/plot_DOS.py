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
H = hh.HubbardHamiltonian(Hsp2)


for u in [0., 3.5]:
    H.U = u
    H.read_density(fn+'/triangulene-AFM.nc')
    dn = H.converge()

    H.find_midgap()
    ev_up = H.eigh(spin=0)
    ev_up -= H.midgap
    ev_dn = H.eigh(spin=1)
    ev_dn -= H.midgap

    p = plot.DOS_distribution(H, E=ev_up[H.Nup], realspace=True)
    p.savefig(fn+'/U%i_DOS.pdf'%(H.U*100))
