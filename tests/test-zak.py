from hubbard import HubbardHamiltonian, sp2
import numpy as np
import sisl

for w in range(1, 25, 2):
    g = sisl.geom.agnr(w)
    H0 = sp2(g)
    H = HubbardHamiltonian(H0, U=0)
    #(self, func=None, nk=51, sub='filled', eigvals=False, method='zak')
    zak = H.get_Zak_phase()
    zako = H.get_Zak_phase(method='zak:origin')
    print(f'width={w:3}, zak={zak:7.3f}, zak:origin={zako:7.3f}')
