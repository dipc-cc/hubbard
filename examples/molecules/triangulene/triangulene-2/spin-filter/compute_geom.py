import Hubbard.geometry as geometry
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('../molecule.xyz').read_geometry()
#mol = mol.move(-mol.center(what='xyz'))
mol.set_sc([3.5*1.42*2*np.sin(np.pi/3), 3*1.42*2*(1+np.cos(np.pi/3)), 10])
mol.sc.set_nsc([1,1,1])

# Select (3,1,w) ribbon
w = 4
gnr, theta = geometry.cgnr(3,1,w, ch_angle=True)
gnr = gnr.tile(2,axis=0)
theta = theta*180/np.pi
gnr = gnr.rotate(theta, v=[0,0,1])
gnr = gnr.mirror('yz')

# Add molecule at the end of the cgnr
gnr1 = gnr.move([-min(mol.xyz[:,0]), -min(mol.xyz[:,1]), 0])
mol = mol.move([5*np.sin(np.pi/3)*2*1.42, 1.42*3*(np.cos(np.pi/2)*2+2),0 ])
gnr1 = gnr1.add(mol)

# Add second cgnr 
gnr2 = gnr.move([12*np.sin(np.pi/3)*2*1.42, 1.42*(2+3*np.cos(np.pi/3)),0 ])
gnr2 = gnr1.add(gnr2)
gnr2.write('3-1-%i-spin-filter.xyz'%w)