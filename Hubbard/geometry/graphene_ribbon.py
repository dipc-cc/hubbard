"""

:mod:`Hubbard.geometry`
==========================

Function to generate graphene-based geometries

.. currentmodule:: Hubbard.geometry

"""

from __future__ import print_function
import numpy as np
import sisl

cs = np.cos(np.pi/3)
sn = np.sin(np.pi/3)

def zgnr(w, bond=1.42):
    
    # ZGNR coordinates
    rA = np.array([0,0,0])
    rB = np.array([sn,cs,0])
    v1 = np.array([sn,1+cs,0])
    v2 = np.array([-sn,1+cs,0])
    xyz = [rA,rB]
    for i in range(w-1):
        if i%2==0:
            xyz.append(xyz[-2]+v1)
            xyz.append(xyz[-2]+v2)
        else:
            xyz.append(xyz[-2]+v2)
            xyz.append(xyz[-2]+v1)
    xyz = bond*np.array(xyz)
    # Cell vectors
    vx = 2.*sn*bond # Periodicity in x-direction (x)
    vy = 2*(1+2*cs)*bond*w # Spacing equal to GNR width (y)
    vz = 10.*bond # 10 bond lengths separation out of GNR plane (z)
    # Create supercell and geometry
    sc = sisl.SuperCell([vx,vy,vz,90,90,90],nsc=[3,1,1])
    uc = sisl.Geometry(list(xyz), atom=sisl.Atom(Z=6,R=bond,orbs=1), sc=sc)
    return uc

def agnr(w, bond=1.42):
 
    # AGNR coordinates
    rA = np.array([0,0,0])
    rB = np.array([1+2*cs,0,0])
    v1 = np.array([1+cs,sn,0])
    v2 = np.array([-1-cs,sn,0])
    xyz = [rA,rB]
    for i in range(w-1):
        if i%2==0:
            xyz.append(xyz[-2]+v1)
            xyz.append(xyz[-2]+v2)
        else:
            xyz.append(xyz[-2]+v2)
            xyz.append(xyz[-2]+v1)
    xyz = bond*np.array(xyz)
    # Cell vectors
    vx = 2.*(1.0+cs)*bond # Periodicity in x-direction (x)
    vy = 2*(2*sn*bond*w) # Spacing equal to GNR width (y)
    vz = 10.*bond # 10 bond lengths separation out of GNR plane (z)
    # Create supercell and geometry
    sc = sisl.SuperCell([vx,vy,vz,90,90,90], nsc=[3,1,1])
    uc = sisl.Geometry(list(xyz), atom=sisl.Atom(Z=6,R=bond,orbs=1), sc=sc)
    return uc

def cgnr(n, m, w, bond=1.42, ch_angle=False):
   
    "Generation of chiral GNR geometry (periodic along x-axis)"
    g = zgnr(w, bond=bond)
    g = g.tile(n+1, axis=0)
    if w %2 == 0 :
        g = g.remove(np.where(g.xyz[:,0] == min(g.xyz[:,0])))
    else:
        natoms = w - (2*m + 1)
        g = g.remove([0,-1])
        g_max = np.where(g.xyz[:,0] == max(g.xyz[:,0]))[0]
        g_min = np.where(g.xyz[:,0] == min(g.xyz[:,0]))[0]
        g = g.remove([list(g_min[:natoms//2])+list(g_max[-natoms//2:])])
    v1 = bond*(1.+cs)*(2.*(m-1)+1.)
    v2 = bond*(n+0.5)*sn*2.
    theta = np.arctan(v1/v2)
    uc = g.rotate(-theta*180/(np.pi), v=[0,0,1])
    uc.set_sc([v2*np.cos(theta)+v1*np.sin(theta), 10, 10])
    uc.set_nsc([3,1,1])
    # Move center-of-mass to origo
    uc = uc.move(-uc.center())

    if ch_angle:
        return uc, theta
    else:
        return uc
