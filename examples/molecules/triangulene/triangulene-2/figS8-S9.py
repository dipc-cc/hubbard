import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl
import funcs

fn = 'dimer-pentagon/'
mol, H = funcs.read(fn, file='molecule.XV')
H.polarize_sublattices()

f = open(fn+'/AFM-FM1-FM2.dat', 'w')

nup_AFM, ndn_AFM = H.nup*1, H.ndn*1
nup_FM1, ndn_FM1 = H.nup*1, H.ndn*1
nup_FM2, ndn_FM2 = H.nup*1, H.ndn*1

for u in np.linspace(5.0, 0.0, 21):

    H.U = u

    # AFM case first
    ncf = fn+'/triangulene-AFM.nc'
    H.nup, H.ndn = nup_AFM, ndn_AFM
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    eAFM = H.Etot
    H.write_density(ncf)
    nup_AFM, ndn_AFM = H.nup*1, H.ndn*1

    if u == 3.0:
        funcs.plot_spin(H, mol, fn+'/AFM-pol-%i.pdf'%(u*100))

    # Now FM1 case
    ncf = fn+'/triangulene-FM1.nc'
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1

    H.nup, H.ndn = nup_FM1, ndn_FM1
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    eFM1 = H.Etot
    H.write_density(ncf)
    nup_FM1, ndn_FM1 = H.nup*1, H.ndn*1

    if u == 3.0:
        funcs.plot_spin(H, mol, fn+'/FM1-pol-%i.pdf'%(u*100))

    # Now FM2 case
    ncf = fn+'/triangulene-FM2.nc'
    H.Nup += 1 # change to four more up-electrons than down
    H.Ndn -= 1

    H.nup, H.ndn = nup_FM2, ndn_FM2
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    eFM2 = H.Etot
    H.write_density(ncf)
    nup_FM2, ndn_FM2 = H.nup*1, H.ndn*1

    if u == 3.0:
        funcs.plot_spin(H, mol, fn+'/FM2-pol-%i.pdf'%(u*100))

    # Revert imbalance
    H.Nup -= 2
    H.Ndn += 2

    f.write('%.4f %.8f %.8f %.8f\n'%(H.U, eAFM, eFM1, eFM2))

f.close()

data = np.loadtxt(fn+'AFM-FM1-FM2.dat')
p = plot.Plot()
p.axes.plot(data[:,0], data[:,1]-data[:,2], 'o', label='S$_{0}$-S$_{1}$')
p.axes.plot(data[:,0], data[:,3]-data[:,2], 'o', label='S$_{2}$-S$_{1}$')
p.set_xlabel(r'U [eV]', fontsize=30)
p.set_xlim(-0.1, 4.1)
p.axes.set_xticks([0,1,2,3,4])
p.axes.set_xticklabels(p.axes.get_xticks(), fontsize=20)
p.axes.set_yticklabels(['%.1f'%i for i in p.axes.get_yticks()], fontsize=20)
p.set_ylabel(r'E$_{S_{i}}$-E$_{S_{0}}$ [eV]', fontsize=25)
p.axes.legend()
p.savefig(fn+'figS9.pdf')

H.polarize_sublattices()

# Plot WF, spectrum and DOS of the GS (in this case FM1)
for u in [0., 3.0]:
    if u > 0:
        H.Nup += 1
        H.Ndn -= 1
        
    H.U = u

    funcs.plot_spectrum(fn, H, mol, '/triangulene-FM1.nc')
    