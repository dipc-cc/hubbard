import Hubbard.hubbard as HH
import sys
import numpy as np

# 3NN tight-binding model
H = HH.Hubbard('junction-2-2.XV', t1=2.7, t2=0.2, t3=.18,
               what='xyz', angle=220)

for u in [0.0, 3.5]:
    # We approach the solutions from above, starting at U=4eV
    H.U = u
    H.read() # Try reading, if we already have density on file
    H.converge()
    H.save() # Computed density to file
    H.plot_wf(EnWindow=0.4, ispin=0)
    H.plot_wf(EnWindow=0.2, ispin=1)
    H.plot_rs_wf(EnWindow=0.4, ispin=0)
    H.plot_rs_wf(EnWindow=0.2, ispin=1)
    H.plot_spectrum(ymax=0.12)
