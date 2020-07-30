from __future__ import print_function
import sisl
import numpy as np
import Hubbard.geometry as geometry
import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2
import Hubbard.plot as plot
import Hubbard.density as density
from Hubbard.negf import NEGF
import os

'''
This script benchmarks the 5-ZGNR constriction of ref. [Y. Hancock, et al., PRB 81, 245402 (2010)]
i.e. the system of fig. 3(a) and the transmissions shown in fig. 4(b) and fig. 4(c)
To describe the system model D of this paper is used (t1=2.7, t2=0.2, t3=0.18, U=2.0 eV)
'''

# Set U and kT for the whole calculation
U = 2.0
kT = 0.025

# Build zigzag GNR
ZGNR = geometry.zgnr(5)

# and 3NN TB Hamiltonian
H_elec = sp2(ZGNR, t1=2.7, t2=0.2, t3=0.18)

mixer = sisl.mixing.PulayMixer(0.6, history=7)

# Hubbard Hamiltonian of elecs
MFH_elec = hh.HubbardHamiltonian(H_elec, U=U, nkpt=[102, 1, 1], kT=kT)
# Initial densities
success = MFH_elec.read_density('elec_density.nc')
if not success:
    # If no densities saved, start with random densities with maximized polarization at the edges
    MFH_elec.random_density()
    MFH_elec.set_polarization([0], dn=[9])

# Converge Electrode Hamiltonians
dn = MFH_elec.converge(density.dm, mixer=mixer)
# Write also densities for future calculations
MFH_elec.write_density('elec_density.nc')
# Plot spin polarization of electrodes
p = plot.SpinPolarization(MFH_elec, colorbar=True)
p.savefig('spin_elecs.pdf')

# Find Fermi level of reservoirs and write to netcdf file
Ef_elecs = MFH_elec.fermi_level(q=MFH_elec.q)
MFH_elec.H.shift(-Ef_elecs)
MFH_elec.H.write('MFH_elec.nc')

# Build central region TB Hamiltonian
HC = H_elec.tile(16, axis=0)
HC = HC.remove([69, 79, 89, 78, 88, 66, 77, 76, 87, 86, 65, 74, 75, 84, 85])
HC.set_nsc([1, 1, 1])
HC.geometry.write('device.xyz')

# Map electrodes in the device region
elec_indx = [range(len(H_elec)), range(len(HC.H)-len(H_elec), len(HC.H))]

# MFH object of the device
MFH_HC = hh.HubbardHamiltonian(HC.H, U=U, kT=kT)
# Initial densities
success = MFH_HC.read_density('HC_density.nc')
if not success:
    # Get initial spin-densities with PBC to speed up the following convergence with OBC
    DM = MFH_elec.tile(16, axis=0).DM
    DM = DM.remove([69, 79, 89, 78, 88, 66, 77, 76, 87, 86, 65, 74, 75, 84, 85])
    MFH_HC.set_dm(DM)

# First create NEGF object
negf = NEGF(MFH_HC, [(MFH_elec, '-A'), (MFH_elec, '+A')], elec_indx, V=0.1)
mixer.clear()
dn = MFH_HC.converge(negf.dm_open, steps=1, tol=1e-5, mixer=mixer, func_args={'qtol': 1e-4}, print_info=True)

print('Nup, Ndn: ', MFH_HC.dm.sum(axis=1))
# Write also densities for future calculations
MFH_HC.write_density('HC_density.nc')
# Plot spin polarization of electrodes
p = plot.SpinPolarization(MFH_HC, colorbar=True)
p.savefig('spin_HC.pdf')

# Shift with Fermi-level of the device and write Hamiltonian into netcdf file
MFH_HC.H.shift(negf.Ef)
MFH_HC.H.write('MFH_HC.nc')

# RUN TBtrans and plot transmissions
print('Clean TBtrans output from previous run')
os.system('rm device.TBT*')
os.system('rm fdf*')
print('Running TBtrans')
os.system('tbtrans RUN.fdf > RUN.out')

tbt_up = sisl.get_sile('device.TBT_UP.nc')
tbt_dn = sisl.get_sile('device.TBT_DN.nc')

p = plot.Plot()
p.axes.plot(tbt_up.E, tbt_up.transmission(0, 1), color='k', label=r'$\sigma=\uparrow$')
p.axes.plot(tbt_dn.E, tbt_dn.transmission(0, 1), '--', color='r', label=r'$\sigma=\downarrow$')
p.axes.legend()
p.set_xlim(-2, 2)
p.set_xlabel('Energy [eV]')
p.set_ylabel('Transmission [a.u.]')
p.savefig('transmission.pdf')
