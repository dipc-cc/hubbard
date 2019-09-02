import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl
import funcs

for i, fn in enumerate(['2H-pos-1-2/', 'pos-1/', 'pos-2/']):
    fn = 'H-passivation/dimer-pentagon/'+fn
    mol, H = funcs.read(fn)

    H.polarize_sublattices()

    nup, ndn = H.nup*1, H.ndn*1

    H.U = 3.0

    ncf = fn+'/triangulene.nc'
    H.nup, H.ndn = nup, ndn
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    e = H.Etot
    H.write_density(ncf)
    nup, ndn = H.nup*1, H.ndn*1

    funcs.plot_spin(H, mol, fn+'/pol-U%i.pdf'%(H.U*100))
    funcs.plot_spectrum(fn, H, mol, fn+'/triangulene.nc')

