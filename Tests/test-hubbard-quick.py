from __future__ import print_function

import Hubbard.hubbard as HH

# Run one iteration
H = HH.Hubbard('mol-ref/mol-ref.XV', U=5.0)
dn, etot = H.iterate(mix=.1)
print(dn, etot)

# Run also one iteration with data from ncfile
H = HH.Hubbard('mol-ref/mol-ref.XV', U=3.5)
dn, etot = H.iterate(mix=1)
print(dn, etot)
