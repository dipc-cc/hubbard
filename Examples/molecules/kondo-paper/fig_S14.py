import Hubbard.hubbard as HH
import sys
import numpy as np

# 3NN tight-binding model
H = HH.Hubbard('junction-2-2.XV', t1=2.7, t2=0.2, t3=.18)

# Output file to collect the energy difference between
# FM and AFM solutions
f = open('FM-AFM.dat', 'w')

for u in np.linspace(0.0, 1.4, 15):
    # We approach the solutions from above, starting at U=4eV
    H.U = 4.4-u
    H.read() # Try reading, if we already have density on file

    # AFM case first
    dn, eAFM = H.converge()
    H.save() # Computed density to file

    # Now FM case
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1
    H.read()
    dn, eFM = H.converge()
    H.save()

    # Revert the imbalance for next loop
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()
