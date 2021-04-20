import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt
import hubbard.density as density
from hubbard.negf import NEGF
import hubbard.hamiltonian as hh
import hubbard.sp2 as sp2

# Set U and kT for the whole calculation
U = 3.
kT = 0.025

# Build zigzag GNR
ZGNR = sisl.geom.zgnr(2)

# and 3NN TB Hamiltonian
H_elec = sp2(ZGNR, t1=2.7, t2=0.2, t3=0.18)

# Hubbard Hamiltonian of elecs
MFH_elec = hh.HubbardHamiltonian(H_elec, U=U, nkpt=[102, 1, 1], kT=kT)

# Start with random densities
MFH_elec.random_density()
# Converge Electrode Hamiltonians
dn = MFH_elec.converge(density.calc_occ, mixer=sisl.mixing.PulayMixer(weight=.7, history=7), tol=1e-10)

# Central region is a repetition of the electrodes without PBC
HC = H_elec.tile(3, axis=0)
HC.set_nsc([1, 1, 1])

# Map electrodes in the device region
elec_indx = [range(len(H_elec)), range(len(HC.H)-len(H_elec), len(HC.H))]

# MFH object
MFH_HC = hh.HubbardHamiltonian(HC.H, DM=MFH_elec.DM.tile(3, axis=0), U=U, kT=kT)

# First create NEGF object
negf = NEGF(MFH_HC, [(MFH_elec, '-A'), (MFH_elec, '+A')], elec_indx)
# Converge using Green's function method to obtain the densities
dn = MFH_HC.converge(negf.calc_occ_open, steps=1, mixer=sisl.mixing.PulayMixer(weight=.1), tol=0.1)
dn = MFH_HC.converge(negf.calc_occ_open, steps=1, mixer=sisl.mixing.PulayMixer(weight=1., history=7), tol=1e-6, print_info=True)

assert abs(MFH_HC.occ[0].sum() - MFH_HC.q[0]) < 1e-5
assert abs(MFH_HC.occ[1].sum() - MFH_HC.q[1]) < 1e-5
print('MFH-NEGF Etot = {:10.5f}'.format(MFH_HC.Etot))

# Reference test for total energy
HC_periodic = H_elec.tile(3, axis=0)
MFH_HC_periodic = hh.HubbardHamiltonian(HC_periodic.H, DM=MFH_elec.DM.tile(3, axis=0), U=U, nkpt=[int(102/3), 1, 1], kT=kT)
dn = MFH_HC_periodic.converge(density.calc_occ)

assert abs(MFH_HC_periodic.occ[0].sum() - MFH_HC_periodic.q[0]) < 1e-7
assert abs(MFH_HC_periodic.occ[1].sum() - MFH_HC_periodic.q[1]) < 1e-7
print('MFH-PER Etot = {:10.5f}'.format(MFH_HC_periodic.Etot))
print('Diff:')
print(MFH_HC_periodic.Etot - MFH_HC.Etot)
