import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('molecule.xyz').read_geometry()
mol = mol.move(-mol.center(what='xyz'))
mol.sc.set_nsc([1,1,1])

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
H = hh.HubbardHamiltonian(Hsp2)
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
        p = plot.SpinPolarization(H, ext_geom=mol, colorbar=True, vmax=0.4)
        #p.annotate()
        p.axes.axis('off')
        p.savefig('AFM-pol-%i.pdf'%(u*100))
        p.close()


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
        p = plot.SpinPolarization(H, ext_geom=mol, colorbar=True, vmax=0.4)
        #p.annotate()
        p.axes.axis('off')
        p.savefig('FM-pol-%i.pdf'%(u*100))
        p.close()


    # Revert imbalance
    H.Nup -= 1
    H.Ndn += 1

    f.write('%.4f %.8f %.8f\n'%(H.U, eAFM, eFM))

f.close()

data = np.loadtxt('AFM-FM.dat')
p = plot.Plot()
p.axes.plot(data[:,0], data[:,2]-data[:,1], 'o', label='S$_{1}$-S$_{0}$')
p.set_xlabel(r'U [eV]', fontsize=30)
p.set_xlim(0,4.1)
p.axes.set_xticks([0,1,2,3])
p.axes.set_xticklabels(p.axes.get_xticks(), fontsize=25)
p.set_ylim(-0.09, 0.01)
p.axes.set_yticks([0.0, -0.02, -0.04, -0.06, -0.08])
p.axes.set_yticklabels(p.axes.get_yticks(), fontsize=25)
p.set_ylabel(r'E$_{FM}$-E$_{AFM}$ [eV]', fontsize=30)
#p.axes.legend()
p.savefig('figS4.pdf')

H.polarize_sublattices()

# Plot WF, spectrum and DOS of the GS (in this case FM)
for u in [0., 3.0]:
    if u > 0:
        H.Nup += 1
        H.Ndn -= 1

    H.U = u

    H.read_density('triangulene-FM.nc')
        
    dn = H.converge()

    H.find_midgap()
    ev_up, evec_up = H.eigh(spin=0, eigvals_only=False)
    ev_up -= H.midgap
    ev_dn, evec_dn = H.eigh(spin=1, eigvals_only=False)
    ev_dn -= H.midgap

    p = plot.Wavefunction(H, evec_up[:, H.Nup-1], ext_geom=mol, realspace=True, vmax=0.001)
    p.axes.set_title(r'$E_{\uparrow}=%.2f$ eV'%(ev_up[H.Nup-1]), fontsize=30, y=-0.15)
    p.axes.axis('off')
    p.savefig('U%i_state%i_up.pdf'%(H.U*100, H.Nup-1))

    p = plot.Wavefunction(H, evec_up[:, H.Nup], ext_geom=mol, realspace=True, vmax=0.001)
    p.axes.set_title(r'$E_{\uparrow}=%.2f$ eV'%(ev_up[H.Nup]), fontsize=30, y=-0.15)
    p.axes.axis('off')
    p.savefig('U%i_state%i_up.pdf'%(H.U*100, H.Nup))

    p = plot.Wavefunction(H, evec_dn[:, H.Ndn-1], ext_geom=mol, realspace=True, vmax=0.001)
    p.axes.set_title(r'$E_{\downarrow}=%.2f$ eV'%(ev_dn[H.Ndn-1]), fontsize=30, y=-0.15)
    p.axes.axis('off')
    p.savefig('U%i_state%i_dn.pdf'%(H.U*100,H.Ndn-1))
    
    p = plot.Wavefunction(H, evec_dn[:, H.Ndn], ext_geom=mol, realspace=True, vmax=0.001)
    p.axes.set_title(r'$E_{\downarrow}=%.2f$ eV'%(ev_dn[H.Ndn]), fontsize=30, y=-0.15)
    p.axes.axis('off')
    p.savefig('U%i_state%i_dn.pdf'%(H.U*100,H.Ndn))

    p = plot.Spectrum(H, ymax=0.13, ymin=0.03)
    p.savefig('U%i_spectrum.pdf'%(H.U*100))
    p.close()

    E = ev_up[H.Nup-1] 
    p = plot.DOS_distribution(H, ext_geom=mol, E=E, realspace=True)
    p.set_title('E $= %.2f$ eV'%(E))
    p.axes.axis('off')
    p.savefig('U%i_DOS.pdf'%(H.U*100))
