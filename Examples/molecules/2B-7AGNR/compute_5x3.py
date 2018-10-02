import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np

H = hh.HubbardHamiltonian('7AGNR2B_5x3.XV', t1=2.7, t2=0.2, t3=.18, U=5.0, what='xyz')
H.polarize_sites([1, 99], dn=[80, 152])
dn, etot = H.converge()
H.save()
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x3.pdf')

"""
H.plot_wf(EnWindow=1.5, density=False)
H.plot_wf(EnWindow=1.5, density=True)
H.plot_rs_wf(EnWindow=1.5, density=False)
H.plot_polarization()
H.plot_charge()
H.plot_rs_polarization()
H.plot_spectrum(xmax=2.0, ymax=0.15)
"""
