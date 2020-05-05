import sisl

graphene = sisl.geom.graphene()

H = sisl.Hamiltonian(graphene)
for ia, io in H:
    idx = H.geom.close(ia, R=[0.1, 1.43])
    H[io, idx[0]] = 0.
    H[io, idx[1]] = -2.7

print(H.eigh(k=[0., 0.5, 0.]))
