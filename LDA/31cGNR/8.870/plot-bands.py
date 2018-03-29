import Hubbard.hubbard as HH

# 3NN
H = HH.Hubbard('molecule.XV', t1=2.7, t2=0.2, t3=.18,
               nsc=[3, 1, 1], kmesh=[100, 1, 1])

H.U = 5.
H.read()

# precondition?
for i in range(1):
    print H.iterate(mix=.1)

for i in range(50):
    print H.iterate(mix=1)

H.save()
H.plot_polarization(f=400)
H.plot_bands()
