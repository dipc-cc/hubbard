import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np

fn = '7AGNR2B_5x3.XV'

grp = 'AFM-AFM-AFM'
H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=5.0, what='xyz', ncgroup=grp)
H.polarize_sites([1, 99], dn=[80, 152])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x3_%s.pdf'%grp)

grp = 'AFM-FM-AFM'
H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=5.0, what='xyz', ncgroup=grp)
H.polarize_sites([1, 152], dn=[80, 99])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x3_%s.pdf'%grp)

grp = 'FM-AFM-FM'
H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=5.0, what='xyz', ncgroup=grp)
H.polarize_sites([1, 80], dn=[99, 152])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x3_%s.pdf'%grp)
