import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.ncdf as ncdf
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('type1.xyz').read_geometry()
mol.sc.set_nsc([1,1,1])
mol = mol.move(-mol.center(what='xyz')).rotate(220, [0,0,1])
# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18)

H = hh.HubbardHamiltonian(Hsp2)

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
try:
    c = ncdf.read('fig3_type1.nc') # Try reading, if we already have density on file
    H.nup, H.ndn = c.nup, c.ndn
except:
    H.random_density()
H.converge()
ncdf.write(H, 'fig3_type1.nc')
p = plot.SpinPolarization(H, vmax=0.20)
p.savefig('fig3_pol.pdf')


