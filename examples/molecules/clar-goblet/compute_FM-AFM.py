from hubbard import HubbardHamiltonian, density, sp2, plot
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('clar-goblet.xyz').read_geometry()
mol.sc.set_nsc([1, 1, 1])
mol = mol.move(-mol.center(what='xyz'))

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18)
H = HubbardHamiltonian(Hsp2)

f = open('FM-AFM.dat', 'w')

for u in np.arange(5, 0, -0.25):
    # We approach the solutions for different U values
    H.U = u

    # AFM case first
    H.set_polarization(up=[6], dn=[28])
    H.read_density('clar-goblet.nc', group='AFM') # Try reading, if we already have density on file

    dn = H.converge(density.calc_n_insulator, tol=1e-10, mixer=sisl.mixing.PulayMixer(0.7, history=7))
    eAFM = H.Etot
    H.write_density('clar-goblet.nc', group='AFM')

    if u == 3.5:
        p = plot.SpinPolarization(H, colorbar=True, vmax=0.4, vmin=-0.4)
        p.savefig('spin_pol_U%i_AFM.pdf'%(H.U*1000))

    # Now FM case
    H.q[0] += 1 # change to two more up-electrons than down
    H.q[1] -= 1

    try:
        H.read_density('clar-goblet.nc', group='FM') # Try reading, if we already have density on file
    except:
        H.set_polarization(up=[6, 28])

    dn = H.converge(density.calc_n_insulator, tol=1e-10, mixer=sisl.mixing.PulayMixer(0.7, history=7))
    eFM = H.Etot
    H.write_density('clar-goblet.nc', group='FM')

    if u == 3.5:
        p = plot.SpinPolarization(H, colorbar=True, vmax=0.4, vmin=-0.4)
        p.savefig('spin_pol_U%i_FM.pdf'%(H.U*1000))

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
