import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np
import sisl

# Build sisl Geometry object
fn = sisl.get_sile('junction-2-2.XV').read_geometry()
fn.sc.set_nsc([1,1,1])
fn = fn.move(-fn.center(what='xyz')).rotate(220, [0,0,1])

# 3NN tight-binding model
H = hh.HubbardHamiltonian(fn, fn_title='junction-2-2', t1=2.7, t2=0.2, t3=.18)

for u in [0.0]:
    # We approach the solutions from above, starting at U=4eV
    H.U = u
    H.read() # Try reading, if we already have density on file
    H.converge()
    H.save() # Computed density to file

    # Plot Eigenspectrum
    p = plot.Spectrum(H, ymax=0.12)
    p.set_title(r'3NN, $U=%.2f$ eV'%H.U)
    p.savefig('eigenspectrum_U%i.pdf'%(H.U*100))

    # Plot HOMO and LUMO level wavefunctions for up- and down-electrons for U=3.5 eV
    spin = ['up', 'dn']
    N = [H.Nup, H.Ndn]
    for i in range(1):
        ev, evec = H.eigh(eigvals_only=False, spin=i)
        ev -= H.midgap
        f = 3800
        v = evec[:,N[i]-1]
        j = np.argmax(abs(v))
        wf = f*v**2*np.sign(v[j])*np.sign(v)
        p = plot.Wavefunction(H, wf)
        p.set_title(r'$E = %.3f$ eV'%(ev[N[i]-1]))
        p.savefig('Fig3_HOMO.pdf')

        v = evec[:,N[i]]
        j = np.argmax(abs(v))
        wf = f*v**2*np.sign(v[j])*np.sign(v)
        p = plot.Wavefunction(H, wf)
        p.set_title(r'$E = %.3f$ eV'%(ev[N[i]]))
        p.savefig('Fig3_LUMO.pdf')

