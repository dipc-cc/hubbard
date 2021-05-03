from hubbard import HubbardHamiltonian, plot, sp2
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('junction-2-2.XV').read_geometry()
mol.sc.set_nsc([1, 1, 1])
mol = mol.move(-mol.center(what='xyz')).rotate(220, [0, 0, 1])
Hsp2 = sp2(mol)

# 3NN tight-binding model
H = HubbardHamiltonian(Hsp2, U=0)

# Plot Eigenspectrum
p = plot.Spectrum(H, ymax=0.12)
p.set_title(r'3NN, $U=%.2f$ eV'%H.U)
p.savefig('eigenspectrum_U%i.pdf'%(H.U*100))

# Plot HOMO and LUMO level wavefunctions for up- and down-electrons for U=3.5 eV
spin = ['up', 'dn']
N = H.q
for i in range(1):
    ev, evec = H.eigh(eigvals_only=False, spin=i)
# Use midgap as energy reference
midgap = H.find_midgap()
ev -= midgap
f = 1
v = evec[:, int(round(N[i]))-1]
j = np.argmax(abs(v))
wf = f*v**2*np.sign(v[j])*np.sign(v)
rs = False # plot in real space?
if not rs:
    wf *= -5000
p = plot.Wavefunction(H, wf, ext_geom=mol, realspace=rs, vmax=0.0006, vmin=-0.0006)
p.set_title(r'$E = %.3f$ eV'%(ev[int(round(N[i]))-1]))
p.axes.axis('off')
p.savefig('Fig3_HOMO.pdf')

v = evec[:, int(round(N[i]))]
j = np.argmax(abs(v))
wf = f*v**2*np.sign(v[j])*np.sign(v)
if not rs:
    wf *= -5000
p = plot.Wavefunction(H, wf, ext_geom=mol, realspace=rs, vmax=0.0006, vmin=-0.0006)
p.set_title(r'$E = %.3f$ eV'%(ev[int(round(N[i]))]))
p.axes.axis('off')
p.savefig('Fig3_LUMO.pdf')
