import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
import sys
import os

U = float(sys.argv[1])

def compute(fn):
    head, tail = os.path.split(fn)
    if '3-1-4' in fn:
        H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=U, nsc=[3, 1, 1], kmesh=[51, 1, 1], what='xyz')
    else:
        H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=U, nsc=[3, 1, 1], kmesh=[51, 1, 1], what='xyz')
    dn, etot = H.iterate()
    if dn > 0.1:
        # We don't have a good solution, try polarizing one edge:
        H.set_polarization([1, 6])
    H.converge(save=True, steps=25)
    H.save()

    p = plot.SpinPolarization(H, colorbar=True, figsize=(6, 10), vmax=0.2)
    p.annotate()
    p.set_title(r'$U=%.2f$ eV [%s]'%(U, head))
    p.savefig(head+'/pol_U%.3i.pdf'%(U*100))

    p = plot.Bandstructure(H)
    p.set_title(r'$U=%.2f$ eV [%s]'%(U, head))
    p.savefig(head+'/bands_U%.3i.pdf'%(U*100))

for fn in sys.argv[2:]:
    compute(fn)
