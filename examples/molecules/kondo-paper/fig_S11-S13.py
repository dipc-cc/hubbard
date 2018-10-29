import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np

# 3NN tight-binding model
H = hh.HubbardHamiltonian('junction-2-2.XV', t1=2.7, t2=0.2, t3=.18,
                          what='xyz', angle=220)

for u in [0.0, 3.5]:
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
    if H.U == 3.5:
        spin = ['up', 'dn']
        N = [H.Nup, H.Ndn]
        for i in range(2):
            ev, evec = H.eigh(eigvals_only=False, spin=i)
            ev -= H.midgap

            p = plot.Wavefunction(H, 1500*evec[:,N[i]-1])
            p.set_title(r'$E = %.3f$ eV'%(ev[N[i]-1]))
            p.savefig('HOMO-%s.pdf'%spin[i])

            p = plot.Wavefunction(H, 1500*evec[:,N[i]])
            p.set_title(r'$E = %.3f$ eV'%(ev[N[i]]))
            p.savefig('LUMO-%s.pdf'%spin[i])

