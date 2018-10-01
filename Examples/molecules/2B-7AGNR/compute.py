import Hubbard.hubbard as HH
import sys
import numpy as np

# 3NN tight-binding model
#H = HH.Hubbard('7AGNR2B_5x5.XV', t1=2.7, t2=0.2, t3=.18, what='xyz')
H = HH.Hubbard('7AGNR2B_5x3.XV', t1=2.7, t2=0.2, t3=.18, what='xyz')

for u in [0.0, 4.0]:
    # We approach the solutions from above, starting at U=4eV
    H.U = u
    H.read() # Try reading, if we already have density on file
    dn, etot = H.converge()
    H.save() # Computed density to file

    H.plot_wf(EnWindow=1.5, density=False)
    H.plot_wf(EnWindow=1.5, density=True)
    H.plot_rs_wf(EnWindow=1.5, density=False)
    H.plot_polarization()
    H.plot_charge()
    H.plot_rs_polarization()
    H.plot_spectrum(xmax=2.0, ymax=0.15)
