import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import Hubbard.ncdf as ncdf
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('7AGNR2B_5x3.XV').read_geometry()
mol.sc.set_nsc([1,1,1])
mol = mol.move(-mol.center(what='xyz'))
H_mol = sp2(mol, t1=2.7, t2=0.2, t3=0.18, dim=2)

pol_up = [[1,99],[1,152],[1,80]]
pol_dn = [[80,152],[80,99],[99,152]]

for ig, grp in enumerate(['AFM-AFM-AFM', 'AFM-FM-AFM', 'FM-AFM-FM']):
    H = hh.HubbardHamiltonian(H_mol, U=5.0)
    fn = '5x3-'+grp
    success = H.read_density(fn+'.nc')
    if not success:
        H.set_polarization(pol_up[ig], dn=pol_dn[ig])
    dn = H.converge()
    H.write_density(fn+'.nc')

    p = plot.SpinPolarization(H)
    p.set_title('Spin Pol 5x3 [%s]'%grp)
    p.annotate()
    p.savefig('%s_pol.pdf'%fn)

    p = plot.LDOSmap(H)
    p.savefig('%s_ldos.pdf'%fn)
