import Hubbard.hamiltonian as hh
import sys
import numpy as np

# 3NN tight-binding model
H = hh.HubbardHamiltonian('2-anthracene.XV', t1=2.7, t2=0.2, t3=.18)

# Output file to collect the energy difference between
# FM and AFM solutions
f = open('FM-AFM.dat', 'w')

for u in np.linspace(0.0, 4.0, 5):
    # We approach the solutions from above, starting at U=4eV
    H.U = 4.0-u

    # AFM case first
    H.read() # Try reading, if we already have density on file
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

    f.write('%.4f %.8f\n'%(4.0-u, eFM-eAFM))

f.close()