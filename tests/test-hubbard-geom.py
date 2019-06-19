import Hubbard.geometry as geom

zgnr = geom.zgnr(8)
zgnr.uc.tile(2, axis=0).write('zgnr.xyz')

agnr = geom.agnr(8)
agnr.uc.tile(2, axis=0).write('agnr.xyz')

geometry = geom.cgnr(3,2,8)
geometry.uc.tile(2,axis=0).write('cgnr.xyz')