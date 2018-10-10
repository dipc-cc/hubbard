import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt

fn = sys.argv[1]

# Read geometry and set up SSH Hamiltonian
geom = sisl.get_sile(fn).read_geom()
geom.set_nsc([3,1,1])
H = sisl.Hamiltonian(geom)
for ia in geom:
    idx = geom.close(ia, R=[0.1, 1.1, 2.1])
    H[ia, idx[0]] = 0.
    H[ia, idx[1]] = 1.0 # 1NN
    H[ia, idx[2]] = 0.5 # 2NN

nx = 1000

# Closed loop, show that this leads to incorrect results
def func(sc, frac):
    return [-0.5+frac, 0, 0]
bzCl = sisl.BrillouinZone(H).parametrize(H, func, nx)
#print bzCl.k

# Open loop, correct integration contour for Zak phase
def func2(sc, frac):
    return [-0.5+1.*nx/(nx-1)*frac, 0, 0]
bzOp = sisl.BrillouinZone(H).parametrize(H, func2, nx)
#print bzOp.k

for band in [0, 1]:
    print '\nBand index =', band
    zak = sisl.electron.berry_phase(bzCl, sub=band, closed=True)
    print 'Zak (closed, incorrect): %.4f rad' % zak
    zak = sisl.electron.berry_phase(bzOp, sub=band, closed=False)
    print 'Zak (open)             : %.4f rad' % zak
    z2 = int(np.abs(1-np.exp(1j*zak))/2)
    print 'Z2 invariant =', z2

if True:
    band = sisl.BandStructure(H, [[0, 0, 0], [0.5, 0, 0]], 100, [r"$\Gamma$", r"$X$"])
    band.set_parent(H)
    bs = band.asarray().eigh()
    lk, kt, kl = band.lineark(True)
    plt.xticks(kt, kl)
    plt.xlim(0, lk[-1])
    plt.ylim([-2, 2])
    plt.ylabel('$E-E_F$ [eV]')
    for bk in bs.T:
        plt.plot(lk, bk)
    plt.savefig('bands.pdf')
