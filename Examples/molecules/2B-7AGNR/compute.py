import Hubbard.hubbard as HH
import sys
import numpy as np

# Density tolerance for quitting the iterations
tol = 1e-10

# 3NN tight-binding model
H = HH.Hubbard('7AGNR2B_5x5.XV', t1=2.7, t2=0.2, t3=.18, what='xyz')
#H = HH.Hubbard('7AGNR2B_5x3.XV', t1=2.7, t2=0.2, t3=.18, what='xyz')

for u in [0.0, 4.0]:
    # We approach the solutions from above, starting at U=4eV
    H.U = u
        
    # AFM case first
    H.read() # Try reading, if we already have density on file

    #H.random_density()

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

    egap, emid = H.find_midgap()
    print egap, emid
    evup = H.Hup.eigh()-emid
    evdn = H.Hdn.eigh()-emid
    for i, e in enumerate(evup):
        if np.abs(e) < 2.0:
            print "%.2i %.4f %.4f"%(i, evup[i], evdn[i])
            
    H.plot_wf(EnWindow=1.5, density=True)
    H.plot_rs_wf(EnWindow=1.5) # Density not acceptable keyword!
    #H.plot_polarization()
    #H.plot_rs_polarization()

    H.plot_localizations()
