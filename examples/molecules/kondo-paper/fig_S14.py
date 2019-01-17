import Hubbard.hamiltonian as hh
import sys
import numpy as np
import sisl

# Build sisl Geometry object
fn = sisl.get_sile('junction-2-2.XV').read_geometry()
fn.sc.set_nsc([1,1,1])

# 3NN tight-binding model
H = hh.HubbardHamiltonian(fn, fn_title='junction-2-2', t1=2.7, t2=0.2, t3=.18)

# Output file to collect the energy difference between
# FM and AFM solutions
f = open('FM-AFM.dat', 'w')

for u in np.linspace(0.0, 1.4, 15):
    # We approach the solutions from above, starting at U=4eV
    H.U = 4.4-u
    H.read() # Try reading, if we already have density on file

    # AFM case first
    dn = H.converge()
    eAFM = H.Etot
    H.save() # Computed density to file

    # Now FM case
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1
    H.read()
    dn = H.converge()
    eFM = H.Etot
    H.save()

    # Revert the imbalance for next loop
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()
