import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np

H = hh.HubbardHamiltonian('7AGNR2B_5x5.XV', t1=2.7, t2=0.2, t3=.18, U=5.0, what='xyz')
H.polarize_sites([3, 99], dn=[82, 178])
dn, etot = H.converge()
H.save()
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x5.pdf')

"""
H.plot_wf(EnWindow=1.5, density=False)
H.plot_wf(EnWindow=1.5, density=True)
H.plot_rs_wf(EnWindow=1.5, density=False)
H.plot_polarization()
H.plot_charge()
H.plot_rs_polarization()
H.plot_spectrum(xmax=2.0, ymax=0.15)
"""
