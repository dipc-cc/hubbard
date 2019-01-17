import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np
import sisl

# Build sisl Geometry object
fn = sisl.get_sile('type1.xyz').read_geometry()
fn.sc.set_nsc([1,1,1])
fn = fn.move(-fn.center(what='xyz')).rotate(220, [0,0,1])

# 3NN tight-binding model
H = hh.HubbardHamiltonian(fn, fn_title='type1', t1=2.7, t2=0.2, t3=.18)

# Plot the single-particle TB (U = 0.0) wavefunction (SO) for Type 1
H.U = 0.0
ev, evec = H.eigh(eigvals_only=False, spin=0)
N = H.Nup
ev -= H.midgap
f = 3800
v = evec[:,N-1]
j = np.argmax(abs(v))
wf = f*v**2*np.sign(v[j])*np.sign(v)
p = plot.Wavefunction(H, wf)
p.set_title(r'$E = %.3f$ eV'%(ev[N-1]))
p.savefig('Fig3_SOMO.pdf')

# Plot MFH spin polarization for U = 3.5 eV
H.U = 3.5
H.read() # Try reading, if we already have density on file
H.converge()
H.save() # Computed density to file
p = plot.SpinPolarization(H, vmax=0.20)
p.savefig('fig3_pol.pdf')


