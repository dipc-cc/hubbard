from __future__ import print_function
import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt

fn = sys.argv[1]

# Read geometry and set up SSH Hamiltonian
geom = sisl.get_sile(fn).read_geometry()
geom = geom.tile(int(sys.argv[2]),axis=0)
geom = geom.move(-geom.center(what='xyz'))
geom.write('test.xyz')
geom.set_nsc([3, 1, 1])
H = sisl.Hamiltonian(geom)
for ia in geom:
    idx = geom.close(ia, R=[0.1, 1.1, 2.1])
    H[ia, idx[0]] = 0.
    H[ia, idx[1]] = 1.0 # 1NN
    H[ia, idx[2]] = 0.5 # 2NN

def func(sc, frac):
    f = [frac, 0, 0]
    return f

# Loop over first two bands, and all occupied ones:
for nk in range(10):
    nx = int(10*1.4**nk)
    bz = sisl.BrillouinZone(H).parametrize(H, func, nx)
    band = range(len(H)//2)
    zak = sisl.electron.berry_phase(bz, sub=band, closed=True, method='Zak')
    z2 = int(np.abs(1-np.exp(1j*zak))/2)
    print(f'Z2={z2} with {nx} k-points')

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
