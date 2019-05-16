import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np

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

def analyze(n, m, w, nx=100):
    geom = cgnr(n, m, w)
    directory = '%i-%i-%i'%(n, m, w)
    H = hh.HubbardHamiltonian(geom, fn_title=directory, t1=2.7, t2=0., t3=0., U=0.0, kmesh=[nx, 1, 1])
    ymax = 2.0
    p = plot.Bandstructure(H, ymax=ymax)
    p.set_title(r'%s'%(directory))
    zak = H.get_Zak_phase(Nx=nx, sub=H.Nup)
    z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
    p.axes.annotate(r'$\gamma=%.4f$'%zak, (0.4, 0.50), size=22, backgroundcolor='w')
    tol = 0.05
    if np.abs(zak) < tol or np.abs(np.abs(zak)-np.pi) < tol:
        # Only append Z2 when appropriate:
        p.axes.annotate(r'$\mathbf{Z_2=%i}$'%z2, (0., 0.9*ymax), size=22, backgroundcolor='k', color='w')
    p.savefig(directory+'/bands_1NN.pdf')

g = analyze(3, 1, 8)

