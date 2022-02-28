from hubbard import HubbardHamiltonian, sp2, plot, density
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol_file = '3-anthracene'
mol = sisl.get_sile(mol_file+'.XV').read_geometry()
mol.sc.set_nsc([1, 1, 1])
mol = mol.move(-mol.center(what='xyz'))

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18)
H = HubbardHamiltonian(Hsp2)

# Output file to collect the energy difference between
# FM and AFM solutions
f = open('FM-AFM.dat', 'w')

mixer = sisl.mixing.PulayMixer(0.7, history=7)

for u in np.linspace(0.0, 4.0, 5):
    # We approach the solutions from above, starting at U=4eV
    H.U = 4.0-u

    # AFM case first
    success = H.read_density(mol_file+'.nc') # Try reading, if we already have density on file
    if not success:
        H.random_density()
        H.set_polarization([1, 6, 15]) # polarize lower zigzag edge
    mixer.clear()
    dn = H.converge(density.calc_n_insulator, mixer=mixer)
    eAFM = H.Etot
    H.write_density(mol_file+'.nc')

    p = plot.SpinPolarization(H, colorbar=True, vmax=0.4, vmin=-0.4)
    p.annotate()
    p.savefig('%s-spin-U%i.pdf'%(mol_file, H.U*1000))

    # Now FM case
    H.q[0] += 1 # change to two more up-electrons than down
    H.q[1] -= 1
    try:
        H.read_density(mol_file+'.nc') # Try reading, if we already have density on file
    except:
        H.random_density()
    mixer.clear()
    dn = H.converge(density.calc_n_insulator, mixer=mixer)
    eFM = H.Etot
    H.write_density(mol_file+'.nc')

    # Revert the imbalance for next loop
    H.q[0] -= 1
    H.q[1] += 1

    f.write('%.4f %.8f\n'%(4.0-u, eFM-eAFM))

f.close()

U, FM_AFM = np.loadtxt('FM-AFM.dat').T
p = plot.Plot()
p.axes.plot(U, FM_AFM, 'o')
p.set_ylabel(r'$E_{FM}-E_{AFM}$ [eV]')
p.set_xlabel(r'$U$ [eV]')
p.savefig('%s-FM_AFM.pdf'%(mol_file))
