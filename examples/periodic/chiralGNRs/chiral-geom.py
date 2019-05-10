import sisl

def cgnr(n, m, w, d=1.42):
    "Generation of chiral GNR geometry (periodic along second lattice vector)"
    g0 = sisl.geom.graphene()
    g = sisl.geom.graphene(orthogonal='True')
    g = g.tile(n+1, 1)
    g = g.remove(3).remove(0)
    g = g.repeat(w/2, 0)
    g.cell[1] += g0.cell[1]
    if m > 1:
        g.cell[1, 0] += 3*(m-1)*d
    g.set_nsc([1,3,1])
    gr = g.repeat(3, 1)
    gr.write('cgnr_%i_%i_%i.xyz'%(n, m, w))
    return g

g = cgnr(3, 1, 8)
print g
