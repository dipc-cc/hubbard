import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np

fn = '7AGNR2B_5x5.XV'

grp = 'AFM-AFM-AFM'
H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=5.0, what='xyz', ncgroup=grp)
H.set_polarization([3, 99], dn=[82, 178])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x5_%s.pdf'%grp)

grp = 'AFM-FM-AFM'
H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=5.0, what='xyz', ncgroup=grp)
H.set_polarization([3, 178], dn=[82, 99])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x5_%s.pdf'%grp)

grp = 'FM-AFM-FM'
H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=5.0, what='xyz', ncgroup=grp)
H.set_polarization([3, 82], dn=[99, 178])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x5_%s.pdf'%grp)
