import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np

# 3NN tight-binding model
H = hh.HubbardHamiltonian('triangulene.xyz', t1=2.7, t2=.2, t3=.18, what='xyz')

f = open('FM-AFM.dat', 'w')

for u in np.linspace(0.0, 3.5, 15):
    # We approach the solutions from above, starting at U=4eV
    H.U = u
    H.read() # Try reading, if we already have density on file

    # AFM case first
    dn, eAFM = H.converge(mix=.5, tol=1e-7)
    H.save() # Computed density to file

    #p = plot.SpinPolarization(H,  colorbar=True)
    #p.annotate()
    #p.savefig('AFM-pol-%i.pdf'%(u*100))

    # Now FM case
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1
    H.read()
    dn, eFM = H.converge(mix=.01, tol=1e-4)
    H.save()

    # Revert the imbalance for next loop
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()
