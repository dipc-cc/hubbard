import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl
import funcs

mol, H = funcs.read('./') 

H.polarize_sublattices()
f = open('AFM-FM.dat', 'w')

nup_AFM, ndn_AFM = H.nup*1, H.ndn*1
nup_FM, ndn_FM = H.nup*1, H.ndn*1

for u in np.linspace(5.0, 0.0, 21):

    H.U = u

    # AFM case first
    ncf = 'triangulene-AFM.nc'
    H.nup, H.ndn = nup_AFM, ndn_AFM
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    eAFM = H.Etot
    H.write_density(ncf)
    nup_AFM, ndn_AFM = H.nup*1, H.ndn*1

    if u == 3.0:
        funcs.plot_spin(H, mol, 'AFM-pol-%i.pdf'%(u*100))


    # Now FM1 case
    ncf = 'triangulene-FM.nc'
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1

    H.nup, H.ndn = nup_FM, ndn_FM
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    eFM = H.Etot
    H.write_density(ncf)
    nup_FM, ndn_FM = H.nup*1, H.ndn*1

    if u == 3.0:
        funcs.plot_spin(H, mol, 'FM-pol-%i.pdf'%(u*100))

    # Revert imbalance
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f %.8f\n'%(H.U, eAFM, eFM))

f.close()

data = np.loadtxt('AFM-FM.dat')
p = plot.Plot()
p.axes.plot(data[:,0], data[:,2]-data[:,1], 'o', label='S$_{1}$-S$_{0}$')
p.set_xlabel(r'U [eV]', fontsize=25)
p.set_xlim(-0.1,4.1)
p.axes.set_xticks([0,1,2,3,4])
p.axes.set_xticklabels(p.axes.get_xticks(), fontsize=20)
p.set_ylim(-0.09, 0.01)
p.axes.set_yticks([0.0, -0.02, -0.04, -0.06, -0.08])
p.axes.set_yticklabels(p.axes.get_yticks(), fontsize=20)
p.set_ylabel(r'E$_{FM}$-E$_{AFM}$ [eV]', fontsize=25)
#p.axes.legend()
p.savefig('figS4.pdf')

H.polarize_sublattices()

# Plot WF, spectrum and DOS of the GS (in this case FM)
for u in [0., 3.0]:
    if u > 0:
        H.Nup += 1
        H.Ndn -= 1

    H.U = u

    funcs.plot_spectrum('./', H, mol, 'triangulene-FM.nc')
