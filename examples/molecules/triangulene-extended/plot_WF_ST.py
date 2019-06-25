import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.ncdf as ncdf
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('triangulene-extended.xyz').read_geometry()
mol = mol.move(-mol.center(what='xyz'))
mol.sc.set_nsc([1,1,1])

for u in [0., 3.5]:

    H = hh.HubbardHamiltonian(mol, t1=2.7, t2=.2, t3=.18, U=u)

    # Plot wavefunctions
    try:
        c = ncdf.read('triangulene.nc', ncgroup='AFM_U%i'%(int(H.U*100))) # Try reading, if we already have density on file
        H.nup, H.ndn = c.nup, c.ndn
    except:
        H.random_density()
    
    dn = H.converge()
    ev_up, evec_up = H.eigh(eigvals_only=False, spin=0)
    ev_up -= H.midgap
    ev_dn, evec_dn = H.eigh(eigvals_only=False, spin=1)
    ev_dn -= H.midgap

    p = plot.Wavefunction(H, 3000*evec_up[:, H.Nup-1])
    p.set_title('$E_{\uparrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_up[H.Nup-1]*1000, H.U))
    p.savefig('U%i_state%i_up.pdf'%(H.U*100, H.Nup-1))

    p = plot.Wavefunction(H, 3000*evec_up[:, H.Nup])
    p.set_title('$E_{\uparrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_up[H.Nup]*1000, H.U))
    p.savefig('U%i_state%i_up.pdf'%(H.U*100, H.Nup))

    p = plot.Wavefunction(H, 3000*evec_up[:, H.Ndn-1])
    p.set_title('$E_{\downarrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_dn[H.Ndn-1]*1000, H.U))
    p.savefig('U%i_state%i_dn.pdf'%(H.U*100,H.Ndn-1))
    
    p = plot.Wavefunction(H, 3000*evec_up[:, H.Ndn])
    p.set_title('$E_{\downarrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_dn[H.Ndn]*1000, H.U))
    p.savefig('U%i_state%i_dn.pdf'%(H.U*100,H.Ndn))

# Plot FM-AFM energies
dat = np.loadtxt('FM-AFM.dat')

p = plot.Plot()
p.axes.plot(dat[:,0], dat[:,1], label='FM-AFM')
p.set_xlabel(r'U [eV]')
p.set_ylabel(r'Energy [eV]')
p.axes.legend()
p.savefig('FM-AFM.pdf')