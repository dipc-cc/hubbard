"""

:mod:`Hubbard.ncdf`
==========================

Function for the meanfield Hubbard Hamiltonian

.. currentmodule:: Hubbard.ncdf

"""

from __future__ import print_function
import numpy as np
import netCDF4 as NC
import hashlib

class read(object):
    '''
    Parameters:
    -----------
    fn : filestr
        Name of netcdf file it is going to read from
    ncgroup : str, optional
        Name of particular ncgroup that is going to be extracted
        
    Note: to follow *exactly* the same calculation, one can use the following 
    identity as a filter: 
        read(fn, ncgroup).hash == ncdf.gethash(H).hash   
    '''
    def __init__(self, fn, ncgroup='default'):

        ncf = NC.Dataset(fn, 'r')
        if ncgroup in ncf.groups:
            print('Reading %s{%s}'%(fn, ncgroup))
            self.ncgroups = ncf.groups
            self.hash = ncf[ncgroup]['hash'][0]
            self.U = ncf[ncgroup]['U'][0]
            self.nup = ncf[ncgroup]['Density'][0][0]
            self.ndn = ncf[ncgroup]['Density'][0][1]
            self.Etot = ncf[ncgroup]['Etot'][0]
            self.Nup = ncf[ncgroup]['Nup'][0]
            self.Ndn = ncf[ncgroup]['Ndn'][0]
        ncf.close()


class write(object):

    def __init__(self, HubbardHamiltonian, fn, ncgroup='default'):
        H = HubbardHamiltonian
        try:
            ncf = NC.Dataset(fn, 'a')
            print('Appending to', fn)
        except:
            print('Initiating', fn)
            ncf = NC.Dataset(fn, 'w')
        g = self.init_ncgrp(H, ncf, ncgroup)
        myhash = gethash(H).hash
        i = np.where(ncf[ncgroup]['hash'][:] == myhash)[0]
        if len(i) == 0:
            i = len(ncf[ncgroup]['hash'][:])
            if g:
                print('Warning! Attempting to save into file that contains different calculation in that ncgroup')
                # Create new ncgroup to avoid overlap between different cacluations
                ncgroup = 'new_group'
                self.init_ncgrp(H, ncf, ncgroup)
        else:
            i = i[0]
        ncf[ncgroup]['hash'][i] = myhash
        ncf[ncgroup]['U'][i] = H.U
        ncf[ncgroup]['Nup'][i] = H.Nup
        ncf[ncgroup]['Ndn'][i] = H.Ndn
        ncf[ncgroup]['Density'][i, 0] = H.nup
        ncf[ncgroup]['Density'][i, 1] = H.ndn
        ncf[ncgroup]['Etot'][i] = H.Etot
        ncf.sync()
        ncf.close()
        print('Wrote (U,Nup,Ndn)=(%.2f,%i,%i) data to %s{%s}'%(H.U, H.Nup, H.Ndn, fn, ncgroup))

        
    def init_ncgrp(self, HubbardHamiltonian, ncf, ncgroup):
        H = HubbardHamiltonian
        if ncgroup not in ncf.groups:
            # create group
            ncf.createGroup(ncgroup)
            ncf[ncgroup].createDimension('unl', None)
            ncf[ncgroup].createDimension('spin', 2)
            ncf[ncgroup].createDimension('sites', len(H.geom))
            ncf[ncgroup].createVariable('hash', 'i8', ('unl',))
            ncf[ncgroup].createVariable('U', 'f8', ('unl',))
            ncf[ncgroup].createVariable('Nup', 'i4', ('unl',))
            ncf[ncgroup].createVariable('Ndn', 'i4', ('unl',))
            ncf[ncgroup].createVariable('Density', 'f8', ('unl', 'spin', 'sites'))
            ncf[ncgroup].createVariable('Etot', 'f8', ('unl',))
            ncf.sync()
        else:
            # Return True in case ncgroup is already an existing group
            return True

class gethash(object):

    def __init__(self, HubbardHamiltonian):
        H = HubbardHamiltonian
        s = ''
        s += 't1=%.2f '%H.t1
        s += 't2=%.2f '%H.t2
        s += 't3=%.2f '%H.t3
        s += 'U=%.2f '%H.U
        s += 'eB=%.2f '%H.eB
        s += 'eN=%.2f '%H.eN
        s += 'Nup=%.2f '%H.Nup
        s += 'Ndn=%.2f '%H.Ndn
        self.hash = int(hashlib.md5(s.encode('utf-8')).hexdigest()[:7], 16)
        self.s = s
