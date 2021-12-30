from sisl import geom, Atom
import os
from add_Hatoms import *
from hubbard import HubbardHamiltonian, sp2, density, plot

W = 7
bond = 1.42

# single-orbital
g = geom.zgnr(W)
TBHam = sp2(g, t1=2.7, t2=0, t3=0)
HH = HubbardHamiltonian(TBHam, U=3, nkpt=[100,1,1])
#print(HH.geometry.atoms.atom[0].q0)
HH.set_polarization([0], dn=[-1])
HH.converge(density.calc_n, print_info=True, tol=1e-10, steps=3)
n_single = HH.n*1

# Start bands-plot, the single-orbital case will be plotted in black
p = plot.Bandstructure(HH, c='k')

# Multi-orbital tight-binding Hamiltonian
pz = sisl.Orbital(1.42, q0=1.0)
s = sisl.Orbital(1.42, q0=0)
C = sisl.Atom(6, orbitals=[pz, s])
g = geom.zgnr(W, atoms=C)

# Add another atom to have also "heteroatom" situation
B = sisl.Atom(6, orbitals=[pz])
G_B = sisl.Geometry(g.xyz[0], atoms=B)
g = g.replace(0,G_B)

# Identify index for atoms
idx = g.a2o(range(len(g)))

# Build U for each orbital in each atom
U = np.zeros(g.no)
U[idx] = 3.

# Build TB Hamiltonian, zeroes for non-pz orbitals
TBham = sisl.Hamiltonian(g, spin='polarized')
for ia in g:
    ib = g.close(ia, R=[0,1.42+0.1])
    io_a = g.a2o(ia, all=True)
    for iib in ib[1]:
        io_b = g.a2o(iib, all=True)
        TBham[io_a[0], io_b[0]] = -2.7

# HubbardHamiltonian object and converge
HH = HubbardHamiltonian(TBham, U=U, nkpt=[100,1,1])
HH.set_polarization([0], dn=[g.a2o(13)])
HH.converge(density.calc_n, print_info=True, tol=1e-10, steps=3)

# Print spin-densities difference compared to sing-orbital case
print('\n   ** Difference between spin densities for single and multi-orbital cases **')
print(HH.n[:,idx]-n_single)

# Add second set of bands for the multi orbital case
p.add_bands(HH, c='--r')
p.savefig('bands.pdf')

# Plot charge for multi-orbital case
p = plot.Charge(HH)
p.savefig('charge.pdf')

# Plot spin polarization for multi-orbital case
p = plot.SpinPolarization(HH, vmax=0.2, vmin=-0.2)
p.savefig('spinpol.pdf')
