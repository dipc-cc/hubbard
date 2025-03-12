from hubbard import HubbardHamiltonian, sp2, density, plot
import numpy as np
import sisl
"""
For this system we get an open-shell solution for U>3 eV
This test obtaines the closed-shell solution for U=2 eV for both a spin-polarized and unpolarized situation
"""

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.set_nsc([1, 1, 1])

Hsp2 = sp2(molecule)

H = HubbardHamiltonian(Hsp2, U=2.0)
H.set_polarization([36],[77])
dn = H.converge(density.calc_n_insulator, mixer=sisl.mixing.LinearMixer(), tol=1e-7)
print('Closed-shell spin-polarized calculation:')
print('dn: {}, Etot: {}\n'.format(dn, H.Etot))

p = plot.Plot()
for i in range(2):
    ev = H.eigh(spin=i)-H.find_midgap()
    ev = ev[abs(ev)<2]
    p.axes.plot(ev, np.zeros_like(ev), ['or', 'xg'][i], label=[r'$\sigma=\uparrow$', r'$\sigma=\downarrow$'][i])

# Compute same system with spin degeneracy
Hsp2 = sp2(molecule, spin='unpolarized')
H = HubbardHamiltonian(Hsp2, U=2.0)
dn = H.converge(density.calc_n_insulator, mixer=sisl.mixing.LinearMixer(), tol=1e-7)
print('Unpolarized calculation:')
print('dn: {}, Etot: {}'.format(dn, H.Etot))

ev = H.eigh()-H.find_midgap()
ev = ev[abs(ev)<2]
p.axes.plot(ev, np.zeros_like(ev), '^k', label='unpol')
p.set_xlabel(r'$E-E_{midgap}$ [eV]')
p.legend()
p.savefig('test-ev.pdf')
