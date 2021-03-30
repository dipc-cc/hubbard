import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import Hubbard.density as density
import numpy as np
import os

""" Script to benchmark the Generalized TB method of Ref. Phys. Rev. B 81, 245402 (2010) """

# Create geometry of the periodic (along x-axis) ribbon
agnr = sisl.geom.agnr(14)
zgnr = sisl.geom.zgnr(16)

lab = ['14-AGNR', '16-ZGNR']
mixer = sisl.mixing.PulayMixer(0.7, history=7)
for i, geom in enumerate([agnr, zgnr]):
    # Build TB Hamiltonian, one can use the parameters from the Ref.
    H0 = sp2(geom, t1=2.7, t2=0.2, t3=0.18, s1=0, s2=0, s3=0)

    # Find self-consistent solution with MFH
    H = hh.HubbardHamiltonian(H0, U=2, nkpt=[100, 1, 1])
    # Start with random densities
    H.random_density()
    mixer.clear()
    dn = H.converge(density.dm, mixer=mixer, print_info=True)

    # Plot banstructure of Hubbard Hamiltonian
    p = plot.Bandstructure(H, ymax=3)
    p.savefig('%s_bands.pdf'%(lab[i]))
    print('\n')
