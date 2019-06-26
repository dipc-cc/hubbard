import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np
import Hubbard.ncdf as ncdf
import sisl

pol = True

# Build sisl Geometry object
mol = sisl.get_sile('7AGNR2B_5x5.XV').read_geometry()
mol.sc.set_nsc([1,1,1])
mol = mol.move(-mol.center(what='xyz'))

pol_up = [[3, 99], [3, 178], [3, 82]]
pol_dn = [[82, 178], [82, 99], [99, 178]]
for ig, grp in enumerate(['AFM-AFM-AFM', 'AFM-FM-AFM', 'FM-AFM-FM']):
    H = hh.HubbardHamiltonian(mol, t1=2.7, t2=0.2, t3=.18, U=5.0)
    try:
        # Try reading from file
        calc = ncdf.read('7AGNR2B_5x5.nc', ncgroup=grp)
        H.nup, H.ndn = calc.nup, calc.ndn
        H.Nup, H.Ndn = calc.Nup, calc.Ndn
    except:
            H.random_density()
            if pol:
                H.set_polarization(pol_up[ig], dn=pol_dn[ig])
    dn = H.converge()
    ncdf.write(H, '7AGNR2B_5x5.nc', ncgroup=grp)
    p = plot.SpinPolarization(H)
    p.set_title('Spin Pol 5x5 [%s]'%grp)
    p.annotate()
    p.savefig('pol_5x5_%s.pdf'%grp)

    p = plot.LDOSmap(H)
    p.savefig('pol_5x5_%s_ldos.pdf'%grp)
