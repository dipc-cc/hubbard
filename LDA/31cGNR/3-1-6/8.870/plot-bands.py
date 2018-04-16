import Hubbard.hubbard as HH

# 3NN
H = HH.Hubbard('molecule.XV', t1=2.7, t2=0.2, t3=.18,
               nsc=[3, 1, 1], kmesh=[100, 1, 1])

H.U = 4.
H.read()
H.plot_bands()

for u in [4., 3., 2., 1., 0.]:
    H.U = u
    H.read()

    # precondition?
    for i in range(1):
        print H.iterate(mix=.1)

    for i in range(100):
        print H.iterate(mix=1)

    H.save()
    #H.plot_polarization(f=400)
    H.plot_bands('molecule.TSHS')
