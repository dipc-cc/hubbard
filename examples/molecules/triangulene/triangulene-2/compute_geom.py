import Hubbard.geometry as geometry
import sys, os
import numpy as np
import sisl

# Copy standard trianguelene molecule and set supercell
mol = sisl.get_sile('../triangulene-1/molecule.xyz').read_geometry()
mol = mol.move(-mol.center(what='xyz'))
mol.set_sc([3.5*1.42*2*np.sin(np.pi/3), 3*1.42*2*(1+np.cos(np.pi/3)), 10])
mol.sc.set_nsc([1,1,1])

# Add last block of zigzag sites to triangulene and write
z = geometry.zgnr(2).tile(4, axis=0)
z = z.remove([0,3])
z = z.move([min(mol.xyz[:,0])-1.42*np.sin(np.pi/3),min(mol.xyz[:,1])-1.42*(2+2*np.cos(np.pi/3)), 0])
mol = mol.add(z)
mol.write('molecule.xyz')

# Make an n repetition ribbon of the molecule
n = 3
mol2 = mol.copy()
if n>1:
    for i in range(1, n):
        mol_rep = mol.move([3.5*1.42*2*np.sin(np.pi/3)*(i), (i)*1.42*(1+np.cos(np.pi/3)), 0])
        mol2 = mol2.add(mol_rep)
        if not os.path.isdir('%i-blocks'%(i+1)):
            os.mkdir('%i-blocks'%(i+1))
        mol2.write('%i-blocks/molecule.xyz'%(i+1))
