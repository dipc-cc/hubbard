import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt
from hubbard import HubbardHamiltonian, density, NEGF, sp2, plot

# Set U for the whole calculation
U = 3.0
kT = 0.025

# Build AGNRs
AGNR7 = sisl.geom.agnr(7)
AGNR9 = sisl.geom.agnr(9)

# Central region is a junction between the 7 and 9 AGNRs
Nrep = 5
AGNR7_rep = AGNR7.tile(Nrep,axis=0).remove(np.arange(-7, 0))
AGNR9_rep = AGNR9.tile(Nrep,axis=0).move(np.array([AGNR7_rep.xyz[-1,0]+1.42,-1.42*np.sin(np.pi/3), 0]))
junction = AGNR7_rep.add(AGNR9_rep)
junction.set_nsc([1,1,1])
HC = sp2(junction)
HC.geometry.write('7-9-AGNR.xyz')
#HC.geometry.write('7-9-AGNR.xyz')
HC.set_nsc([1, 1, 1])

# and 3NN TB Hamiltonian
H_elec_7 = sp2(AGNR7, t1=2.7, t2=0.2, t3=0.18)
H_elec_9 = sp2(AGNR9, t1=2.7, t2=0.2, t3=0.18)

# Hubbard Hamiltonian of elecs
MFH_elec_7 = HubbardHamiltonian(H_elec_7, U=U, nkpt=[102, 1, 1], kT=kT)
MFH_elec_9 = HubbardHamiltonian(H_elec_9, U=U, nkpt=[102, 1, 1], kT=kT)

# Initialize spin densities
MFH_elec_7.set_polarization(range(int(7/2)), dn=range(-int(7/2), 0)) # Ensure we break symmetry
MFH_elec_9.set_polarization(range(int(9/2)), dn=range(-int(9/2), 0)) # Ensure we break symmetry

# Converge Electrode Hamiltonians
dn = MFH_elec_7.converge(density.calc_n, mixer=sisl.mixing.PulayMixer(weight=.7, history=7), tol=1e-10)
dn = MFH_elec_9.converge(density.calc_n, mixer=sisl.mixing.PulayMixer(weight=.7, history=7), tol=1e-10)

p = plot.SpinPolarization(MFH_elec_7, colorbar=True, vmax=0.2, vmin=-0.2)
p.annotate()
p.savefig('ELEC_AGNR7.pdf')

p = plot.SpinPolarization(MFH_elec_9, colorbar=True, vmax=0.2, vmin=-0.2)
p.annotate()
p.savefig('ELEC_AGNR9.pdf')

dist = sisl.get_distribution('fermi_dirac', smearing=kT)
Ef_7 = MFH_elec_7.fermi_level()
print("Electrode 7-AGNR Ef = ", Ef_7, ' eV')
# Shift each electrode with its Fermi-level and write it to netcdf file
MFH_elec_7.shift(-Ef_7)

Ef_9 = MFH_elec_9.fermi_level()
print("Electrode 9-AGNR Ef = ", Ef_9, ' eV')
# Shift each electrode with its Fermi-level and write it to netcdf file
MFH_elec_9.shift(-Ef_9)

# Map electrodes in the device region
elec_indx = [range(len(H_elec_7)), range(-len(H_elec_9), 0)]

# MFH object
MFH_HC = HubbardHamiltonian(HC, U=U, kT=kT)
MFH_HC.set_polarization([59,62])
print(MFH_HC.q)

# First create NEGF object
negf = NEGF(MFH_HC, [(MFH_elec_7, '-A'), (MFH_elec_9, '+A')], elec_indx)
# Converge using Green's function method to obtain the densities
dn = MFH_HC.converge(negf.calc_n_open, steps=1, mixer=sisl.mixing.PulayMixer(weight=.1), tol=0.1)
dn = MFH_HC.converge(negf.calc_n_open, steps=1, mixer=sisl.mixing.PulayMixer(weight=1., history=7), tol=1e-6, print_info=True)
print('Nup, Ndn, Ntot: ', MFH_HC.n.sum(axis=1), MFH_HC.n.sum())
print('Device potential:', negf.Ef, ' eV')

p = plot.SpinPolarization(MFH_HC, colorbar=True, vmax=0.2, vmin=-0.2)
p.annotate()
p.savefig('junction.pdf')
