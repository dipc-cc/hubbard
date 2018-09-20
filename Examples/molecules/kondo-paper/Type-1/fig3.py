import Hubbard.hubbard as HH
import sys
import numpy as np

# Density tolerance for quitting the iterations
tol = 1e-10

# 3NN tight-binding model
H = HH.Hubbard('type1.xyz', t1=2.7, t2=0.2, t3=.18,
               what='xyz', angle=220)

for u in [0.0, 3.5]:
    # We approach the solutions from above, starting at U=4eV
    H.U = u
    H.read() # Try reading, if we already have density on file

    deltaN = 1.
    i = 0
    while deltaN > tol:
        if deltaN > 0.1:
            # precondition
            deltaN, eAFM = H.iterate(mix=.1)
        else:
            deltaN, eAFM = H.iterate(mix=1)
        i += 1
        if i%100 == 0:
            print "   AFM iteration %i: deltaN = %.8f" %(i, deltaN)
    print "   Converged in %i iterations" %i
    H.save() # Computed density to file
    H.plot_polarization()
    H.plot_wf(EnWindow=0.4, ispin=0)
    H.plot_wf(EnWindow=0.2, ispin=1)
    H.plot_rs_wf(EnWindow=0.4, ispin=0,z=0.8)
    H.plot_rs_wf(EnWindow=0.2, ispin=1,z=0.8)



