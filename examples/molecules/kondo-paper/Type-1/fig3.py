from hubbard import HubbardHamiltonian, plot, density, sp2
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('type1.xyz').read_geometry()
mol.sc.set_nsc([1, 1, 1])
mol = mol.move(-mol.center(what='xyz')).rotate(220, [0, 0, 1])
# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18)

H = HubbardHamiltonian(Hsp2)

# Plot the single-particle TB (U = 0.0) wavefunction (SO) for Type 1
H.U = 0.0
ev, evec = H.eigh(eigvals_only=False, spin=0)
N = H.q[0]
midgap = H.find_midgap()
ev -= midgap
f = 3800
v = evec[:, int(round(N))-1]
j = np.argmax(abs(v))
wf = f*v**2*np.sign(v[j])*np.sign(v)
p = plot.Wavefunction(H, wf)
p.set_title(r'$E = %.3f$ eV'%(ev[int(round(N))-1]))
p.savefig('Fig3_SOMO.pdf')

# Plot MFH spin polarization for U = 3.5 eV
H.U = 3.5
success = H.read_density('fig3_type1.nc') # Try reading, if we already have density on file
if not success:
    H.set_polarization([23])
mixer = sisl.mixing.PulayMixer(0.7, history=7)
H.converge(density.calc_n_insulator, mixer=mixer)
H.write_density('fig3_type1.nc')
p = plot.SpinPolarization(H, ext_geom=mol, vmax=0.20)
p.savefig('fig3_pol.pdf')
