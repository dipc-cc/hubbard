import Hubbard.hubbard as HH



H = HH.Hubbard('../Examples/31cGNR/LDA/3-1-3/8.870/molecule.XV')

H.U = 5.
H.random_density()

print H.iterate(mix=.1)