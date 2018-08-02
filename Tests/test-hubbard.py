import Hubbard.hubbard as HH



H = HH.Hubbard('../Examples/molecules/antracenes/2-antracene.XV')

H.U = 5.
H.random_density()

print H.iterate(mix=.1)
