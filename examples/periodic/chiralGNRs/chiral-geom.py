import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
import os

def cgnr(n, m, w, d=1.42):
    "Generation of chiral GNR geometry (periodic along x-axis)"
    g0 = sisl.geom.graphene()
    g = sisl.geom.graphene(orthogonal='True')
    g = g.tile(n+1, 1)
    g = g.remove(3).remove(0)
    g = g.repeat(w/2, 0)
    g.cell[1] += g0.cell[1]
    if m > 1:
        g.cell[1, 0] += 3*(m-1)*d
    cs = np.cos(np.pi/3)
    sn = np.sin(np.pi/3)
    A1 = d*(1.+cs)*(2.*(m-1)+1.)
    A2 = d*(n+0.5)*sn*2.
    theta = np.arctan(A2/A1)
    gr = g.rotate(theta*360/(2*np.pi), v=[0,0,1])
    gr.set_sc([A2*np.sin(theta)+A1*np.cos(theta), 10, 10])
    gr.set_nsc([3,1,1])
    # Mover center-of-mass to origo
    gr = gr.move(-gr.center())
    return gr

def analyze(n, m, w, nx=1001):
    directory = '%i-%i-%i'%(n, m, w)
    print('Doing', directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    geom = cgnr(n, m, w)
    geom.write(directory+'/cgnr.xyz')
    geom.repeat(3, 0).write(directory+'/cgnr-rep.xyz')
    H = hh.HubbardHamiltonian(geom, fn_title=directory, t1=2.7, t2=0., t3=0., U=0.0, kmesh=[nx, 1, 1])
    ymax = 8.0
    p = plot.Bandstructure(H, ymax=ymax)
    p.set_title(r'%s: $n_x=%i$'%(directory,nx))
    # Zak one band only
    zak1 = H.get_Zak_phase(Nx=nx, sub=H.Nup)
    z21 = int(round(np.abs(1-np.exp(1j*zak1))/2))
    zak1b = H.get_Zak_phase(Nx=nx+1, sub=H.Nup)
    z21b = int(round(np.abs(1-np.exp(1j*zak1b))/2))
    assert z21 == z21b
    # Zak all filled bands
    zak = H.get_Zak_phase(Nx=nx)
    z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
    print('%s: z21=%i [z2-all=%i]'%(directory, z21,z2))
    #assert z21 == z2
    #p.axes.annotate(r'$\gamma=%.4f$'%zak, (0.4, 0.50), size=22, backgroundcolor='w')
    tol = 0.05
    if np.abs(zak) < tol or np.abs(np.abs(zak)-np.pi) < tol:
        # Only append Z2 when appropriate:
        p.axes.annotate(r'$\mathbf{Z_2=%i (%i)}$'%(z21, z2), (0., 0.9*ymax), size=22, backgroundcolor='k', color='w')
    p.savefig(directory+'/bands_1NN.pdf')


def analyze_edge(n,m,w):
    directory = '%i-%i-%i'%(n, m, w)
    print('Doing', directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # Create 15 length ribbon
    geom = cgnr(n,m,w).tile(15, axis=0)
    # Identify edge sites along the lower ribbon border
    geom.set_nsc([3,1,1]) # Set periodic boundary conditions to avoid lateral edge sites
    sites = []
    for ia in geom:
        idx = geom.close(ia, R=[0.1, 1.43])
        if len(idx[1]) == 2 :
            if geom.xyz[ia, 1] < 0:
                sites.append(ia)

    # Eigenvectors and eigenvalues in 1NN model for finite ribbon
    geom.set_nsc([1,1,1])
    H = hh.HubbardHamiltonian(geom, t1=2.7, t2=0., t3=0., U=0.)        
    ev, evec = H.H.eigh(eigvals_only=False,spin=0)
    ev -= H.midgap

    p = plot.Plot()
    y1 = np.absolute(evec[sites, H.Nup-1] )**2
    y2 = np.absolute(evec[sites, H.Nup] )**2
    x = geom.xyz[sites, 0]
    p.axes.plot(x, y1, '-or', label=r'HOMO')
    p.axes.plot(x, y2, '--ob', label=r'LUMO')
    p.axes.legend(fontsize=13)
    p.set_ylabel(r'$|\Psi_{n}(x_{edge})|^{2}$ [a.u.]')
    p.set_xlabel(r'$x$ [\AA]')
    p.set_title('[%s]'%directory)
    p.savefig(directory+'/1NN_squared_wf.pdf')

    if True:
        # Plot edge sites?
        v = np.zeros(len(H.geom))
        v[sites] = 1.
        p = plot.GeometryPlot(H, cmap='Reds', figsize=(10,3))
        p.__orbitals__(v, vmax=1.0, vmin=0)
        p.set_title(r'Edge sites of [%s]'%directory)
        p.savefig(directory+'/edge_sites.pdf')

n = 3
m = 1
g = analyze(n, m, 4)
g = analyze(n, m, 6)
g = analyze(n, m, 8)
g = analyze(n, m, 10)
analyze_edge(n, m, 6)
