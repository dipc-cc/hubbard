from __future__ import print_function

import Hubbard.hubbard as HH

H = HH.Hubbard('../Examples/molecules/anthracenes/2-anthracene.XV')

H.U = 5.
H.random_density()
dn, etot = H.iterate(mix=.1)

print(dn, etot)
