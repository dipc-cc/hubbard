import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2
import Hubbard.plot as plot
import sys
import numpy as np
import Hubbard.density as dm
import sisl

# Build sisl Geometry object
mol_file = '2-anthracene'
mol = sisl.get_sile(mol_file+'.XV').read_geometry()
mol.sc.set_nsc([1,1,1])
mol = mol.move(-mol.center(what='xyz'))

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
H = hh.HubbardHamiltonian(Hsp2)

# Output file to collect the energy difference between
# FM and AFM solutions
f = open('FM-AFM.dat', 'w')

for u in np.linspace(0.0, 4.0, 5):
    # We approach the solutions from above, starting at U=4eV
    H.U = 4.0-u

    # AFM case first
    try:
        H.read_density(mol_file+'.nc') # Try reading, if we already have density on file
    except:
        H.random_density()
    dn = H.converge(dm.dm_insulator)
    eAFM = H.Etot
    H.write_density(mol_file+'.nc')
    
    p = plot.SpinPolarization(H, colorbar=True, vmax=0.4, vmin=-0.4)
    p.savefig('%s-spin-U%i'%(mol_file, H.U*1000))

    # Now FM case
    H.q[0] += 1 # change to two more up-electrons than down
    H.q[1] -= 1
    try:
        H.read_density(mol_file+'.nc') # Try reading, if we already have density on file
    except:
        H.random_density()
    dn = H.converge(dm.dm_insulator)
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
