import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np
import Hubbard.ncdf as ncdf
import sisl

pol = False

# Build sisl Geometry object
mol = sisl.get_sile('7AGNR2B_5x3.XV').read_geometry()
mol.sc.set_nsc([1,1,1])
mol = mol.move(-mol.center(what='xyz'))

up = [[1,99],[1,152],[1,80]]
dn = [[80,152],[80,99],[99,152]]
for ig, grp in enumerate(['AFM-AFM-AFM', 'AFM-FM-AFM', 'FM-AFM-FM']):
    print(grp)
    H = hh.HubbardHamiltonian(mol, t1=2.7, t2=0.2, t3=.18, U=5.0)
    try:
        # Try reading from file
        calc = ncdf.read('7AGNR2B_5x3.nc', ncgroup=grp)
        H.nup, H.ndn = calc.nup, calc.ndn
        H.Nup, H.Ndn = calc.Nup, calc.Ndn
    except:
        if pol:
            H.set_polarization(up[ig], dn=dn[ig])
        else:
            H.random_density()
    dn = H.converge()
    b = ncdf.write(H, '7AGNR2B_5x3.nc', ncgroup=grp)
    p = plot.SpinPolarization(H)
    p.set_title('Spin Pol 5x3 [%s]'%grp)
    p.annotate()
    p.savefig('pol_5x3_%s.pdf'%grp)

    p = plot.LDOSmap(H)
    p.savefig('pol_5x3_%s_ldos.pdf'%grp)
