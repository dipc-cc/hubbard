import Hubbard.hubbard as HH



H = HH.Hubbard('../Examples/molecules/anthracenes/2-anthracene.XV')

H.U = 5.
H.random_density()

print H.iterate(mix=.1)
