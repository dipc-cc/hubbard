import Hubbard.hamiltonian as hh
import sys
import numpy as np

# Density tolerance for quitting the iterations
tol = 1e-10

# 3NN tight-binding model
H = hh.HubbardHamiltonian('type1.xyz', t1=2.7, t2=0.2, t3=.18,
                          what='xyz', angle=220)

for u in [0.0, 3.5]:
    # We approach the solutions from above, starting at U=4eV
    H.U = u
    H.read() # Try reading, if we already have density on file
    H.converge()
    H.save() # Computed density to file
    # The following needs to be updated:
    #H.plot_polarization()
    #H.plot_wf(EnWindow=0.4, ispin=0)
    #H.plot_wf(EnWindow=0.2, ispin=1)
    #H.plot_rs_wf(EnWindow=0.4, ispin=0, z=0.8)
    #H.plot_rs_wf(EnWindow=0.2, ispin=1, z=0.8)
