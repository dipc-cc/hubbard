import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
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

if 'triangulene-1' in fn or 'triangulene-2' in fn:
    H.Nup += 1
    H.Ndn -= 1
    fn_nc = fn+'triangulene-FM.nc'
else:
    fn_nc = fn+'trianglene-AFM.nc'

H.polarize_sublattices()

for u in [0., 3.5]:
    H.U = u

    H.read_density(fn_nc)
        
    dn = H.converge()

    H.find_midgap()
    ev_up = H.eigh(spin=0)
    ev_up -= H.midgap
    ev_dn = H.eigh(spin=1)
    ev_dn -= H.midgap

    E = ev_up[H.Nup-2] 
    p = plot.DOS_distribution(H, ext_geom=mol, E=E, realspace=True)
    p.set_title('E $= %.2f$ eV'%(E))
    p.savefig(fn+'/U%i_DOS.pdf'%(H.U*100))
