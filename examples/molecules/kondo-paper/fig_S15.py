from hubbard import HubbardHamiltonian, density, plot, sp2
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('junction-2-2.XV').read_geometry()
mol.sc.set_nsc([1, 1, 1])
# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18)

H = HubbardHamiltonian(Hsp2)

# Output file to collect the energy difference between
# FM and AFM solutions
f = open('FM-AFM.dat', 'w')

H.set_polarization([77], dn=[23])
for u in np.linspace(0.0, 1.4, 15):
    # We approach the solutions from above, starting at U=4eV
    H.U = 4.4-u
    # AFM case first
    success = H.read_density('fig_S15.nc') # Try reading, if we already have density on file

    dn = H.converge(density.calc_n_insulator, mixer=sisl.mixing.PulayMixer(0.7, history=7), tol=1e-6)
    eAFM = H.Etot
    H.write_density('fig_S15.nc')

    # Now FM case
    H.q[0] += 1 # change to two more up-electrons than down
    H.q[1] -= 1
    success = H.read_density('fig_S15.nc') # Try reading, if we already have density on file

    dn = H.converge(density.calc_n_insulator, mixer=sisl.mixing.PulayMixer(0.7, history=7), tol=1e-6)
    eFM = H.Etot
    H.write_density('fig_S15.nc')

    # Revert the imbalance for next loop
    H.q[0] -= 1
    H.q[1] += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()

U, FM_AFM = np.loadtxt('FM-AFM.dat').T
p = plot.Plot()
p.axes.plot(U, FM_AFM, 'o')
p.set_ylabel(r'$E_{FM}-E_{AFM}$ [eV]')
p.set_xlabel(r'$U$ [eV]')
p.savefig('FM_AFM.pdf')
