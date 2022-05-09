import numpy as np
from scipy import sparse as sp
from numba import njit, jit, prange
import gc
from scipy.linalg import sqrtm

__all__ = ["block_linalg"]

Inv   = np.linalg.inv
Solve = np.linalg.solve
MM    = np.matmul
Wh    = np.where

def error():
    assert 1==0

def test_partition_2d_sparse_matrix(A,P,tol = 1e-5,sparse=True, only_slices = False):
    assert P[-1]==A.shape[0]==A.shape[1]
    
    if sparse==True:
        i,j,v = sp.find(A)
        ABS=np.abs(v).sum()
        N = len(P)
        Slices_a = [ [slice(P[i],P[i+1]),slice(P[i],P[i+1])] for i in range(N-1)   ]
        Slices_b = [ [slice(P[i],P[i+1]),slice(P[i-1],P[i])] for i in range(1,N-1) ]
        Slices_c = [ [slice(P[i-1],P[i]),slice(P[i],P[i+1])] for i in range(1,N-1) ]
        if only_slices:
            return 1.0, [Slices_a,Slices_b,Slices_c]
        
        Abs = 0.0
        
        
        for i in range(N-1):
            Abs+=np.abs(A[Slices_a[i][0],Slices_a[i][1]].todense()).sum()
            if i<N-2:
                Abs+=np.abs(A[Slices_b[i][0],Slices_b[i][1]].todense()).sum()
                Abs+=np.abs(A[Slices_c[i][0],Slices_c[i][1]].todense()).sum()
        
        return Abs/ABS,[Slices_a,Slices_b,Slices_c]

def Transpose(A):
    s=A.shape
    if isinstance(A, np.ndarray):
        if len(s)==2:
            return A.T
        elif len(s)==3:
            return A.transpose(0,2,1)
        elif len(s)==4:
            return A.transpose(0,1,3,2)
        elif len(s)==5:
            return A.transpose(0,1,2,4,3)
    

def slices_to_npslices(Slices):
    N = len(Slices[0])
    np_slices = np.zeros((3,N, 2,2),dtype = np.int32)
    Sa = Slices[0]
    Sb = Slices[1]
    Sc = Slices[2]
    
    for i in range(N):
        ia = Sa[i][0]; ja = Sa[i][1]
        np_slices[0,i,0,:]  = ia.start,ia.stop
        np_slices[0,i,1,:]  = ja.start,ja.stop
        if i<N-1:
            ib = Sb[i][0]; jb = Sb[i][1]
            ic = Sc[i][0]; jc = Sc[i][1]
            np_slices[1,i,0,:]  = ib.start,ib.stop
            np_slices[1,i,1,:]  = jb.start,jb.stop
            np_slices[2,i,0,:]  = ic.start,ic.stop
            np_slices[2,i,1,:]  = jc.start,jc.stop
    return np_slices

@njit(cache = True)
def Build_BTD_purenp(I, J, V, Slices):
    #Input from scipy.sparse.find(A), together with the way the matrix is partitioned into blocks
    # Al = List(); Bl =List(); Cl = List()
    Al = []; Bl = []; Cl = []
    Al.append(np.zeros((1,1), dtype = V.dtype))
    Bl.append(np.zeros((1,1), dtype = V.dtype))
    Cl.append(np.zeros((1,1), dtype = V.dtype))
    
    Ia = []; Ib = []; Ic = []
    
    Ia.append(np.int(1))
    Ib.append(np.int(1))
    Ic.append(np.int(1))
    
    N  = len(Slices[0])
    Sa = Slices[0]
    Sb = Slices[1]
    Sc = Slices[2]
    dt = V.dtype 
    
    for i in range(N):
        
        ia = Sa[i,0]; ja = Sa[i,1]
        ia_start,ia_stop = ia[0], ia[1] 
        ja_start,ja_stop = ja[0], ja[1] 
        
        a = np.zeros((ia_stop-ia_start,ja_stop-ja_start),dtype = dt)
        inds_a  = np.where( (I>=ia_start) * (I<ia_stop) * (J>=ja_start) * (J<ja_stop) )#[0]
        
        li = I[inds_a]-ia_start
        lj = J[inds_a]-ja_start
        v  = V[inds_a]
        
        for k  in range(len(li)):
            a[li[k], lj[k]] = v[k]
        
        Al.append( a )
        
        if i<N-1:
            ib = Sb[i, 0]; jb = Sb[i, 1]
            ic = Sc[i, 0]; jc = Sc[i, 1]
            
            ib_start,ib_stop = ib[0], ib[1] 
            jb_start,jb_stop = jb[0], jb[1] 
            ic_start,ic_stop = ic[0], ic[1] 
            jc_start,jc_stop = jc[0], jc[1] 
            
            b = np.zeros((ib_stop-ib_start,jb_stop-jb_start),dtype = dt)
            inds_b =  np.where( (I>=ib_start) * (I<ib_stop) * (J>=jb_start) * (J<jb_stop) )#[0]
            li_b = I[inds_b]-ib_start
            lj_b = J[inds_b]-jb_start
            v_b  = V[inds_b]
            
            c = np.zeros((ic_stop-ic_start,jc_stop-jc_start),dtype = dt)
            inds_c = np.where( (I>=ic_start) * (I<ic_stop) * (J>=jc_start) * (J<jc_stop) )#[0]
            li_c = I[inds_c]-ic_start
            lj_c = J[inds_c]-jc_start
            v_c  = V[inds_c]
            
            for k in range(len(li_b)):
                b[li_b[k], lj_b[k]] = v_b[k]
                c[li_c[k], lj_c[k]] = v_c[k]
                
            Bl.append( b )
            
            Cl.append( c )
    
    return Al[1:], Bl[1:], Cl[1:]

@njit(parallel = True, cache = True)
def _Build_BTD_vectorised(Iv, Jv, Vv, Slices):
    nv = Iv.shape[0]
    dt = Vv.dtype
    # Alv = List(); Blv =List(); Clv = List()
    Alv = []; Blv = []; Clv = [];
    Alv.append(np.zeros((1,1,1), dtype = dt))
    Blv.append(np.zeros((1,1,1), dtype = dt))
    Clv.append(np.zeros((1,1,1), dtype = dt))
    
    N  = len(Slices[0])
    Sa = Slices[0]
    Sb = Slices[1]
    Sc = Slices[2]
     
    
    for i in range(N):
        ia = Sa[i,0]; ja = Sa[i,1]
        ia_start,ia_stop = ia[0], ia[1] 
        ja_start,ja_stop = ja[0], ja[1] 
        
        a = np.zeros((nv, ia_stop-ia_start,ja_stop-ja_start),dtype = dt)
        Alv.append( a )
        if i<N-1:
            ib = Sb[i, 0]; jb = Sb[i, 1]
            ic = Sc[i, 0]; jc = Sc[i, 1]
            
            ib_start,ib_stop = ib[0], ib[1] 
            jb_start,jb_stop = jb[0], jb[1] 
            ic_start,ic_stop = ic[0], ic[1] 
            jc_start,jc_stop = jc[0], jc[1]
            
            b = np.zeros((nv, ib_stop-ib_start,jb_stop-jb_start),dtype = dt)
            Blv.append( b )
            
            c = np.zeros((nv, ic_stop-ic_start,jc_stop-jc_start),dtype = dt)
            Clv.append( c )
    Alv = Alv[1:]
    Blv = Blv[1:]
    Clv = Clv[1:]
    
    for i in prange(nv):
        A ,B, C = Build_BTD_purenp(Iv[i], Jv[i], Vv[i], Slices)
        for j in range(N):
            Alv[j][i,:,:] = A[j]
            if j < N-1:
                Blv[j][i,:,:] = B[j]
                Clv[j][i,:,:] = C[j]
    
    return Alv, Blv, Clv

def Build_BTD_vectorised(Iv, Jv, Vv, Slices):
    #o1,o2,o3 = _Build_BTD_vectorised(Iv, Jv, Vv, Slices)
    return _Build_BTD_vectorised(Iv, Jv, Vv, Slices)

def sparse_find_faster(A):
    B = A.tocoo()
    return B.row, B.col, B.data

def Find_copies(L,atol=1e-5,rtol = 1e-5):
    def equal(a,b):
        return np.isclose(a,b,atol = atol , rtol = rtol).all()
    
    n = len(L)
    Truth = np.zeros((n,n),dtype=bool)
    for i in range(n):
        for j in range(i+1,n):
            if equal(L[i],L[j]):
                Truth[i,j] = 1

def all_zero(A):
    if A is None:
        return True
    else:
        return not np.any(A)




class block_td:
    ##### Articles used: 
    ##### Matthew G Reuter and Judith C Hill
    ##### An efficient, block-by-block algorithm for inverting
    ##### a block tridiagonal, nearly block Toeplitz matrix
    #####
    ##### And
    #####
    ##### Improvements on non-equilibrium and transport 
    ##### Green function techniques: The next-generation transiesta
    
    #Block Tridiagonal  matrix of numpy arrays (vectorised in the first three indecies)#
    
    def __init__(self,Al,Bl,Cl,I_al,I_bl,I_cl,diagonal_zeros=False,E_grid = None):
        #Matrix-elements and shortcuts to their position in a list
        self.Al = [a.copy() for a in Al]
        self.Bl = [b.copy() for b in Bl]
        self.Cl = [c.copy() for c in Cl]
        self.I_al = I_al.copy()
        self.I_bl = I_bl.copy()
        self.I_cl = I_cl.copy()
        #Sanity checks
        assert len(I_bl)==len(I_cl)
        assert len(I_al)-1==len(I_cl)
        #useful numbers
        self.N=len(I_al)
        self.dt=Al[0].dtype
        self.num_vect_inds = len(Al[0].shape) - 2
        
        self.has_been_conjugated = False
        self.has_been_transposed = False
        
        #nonzero elements and check for block structure
        self.info(diagonal_zeros)
        self.Shape()
        #initialisations
        self.diagonal_zeros=diagonal_zeros
        self.E_grid = E_grid
        ######
    
    def Find_Duplicates(self):
        #Thorough testing not done on this function
        self.Al, self.I_al = find_unique_blocks_and_indecies(self.Al,self.I_al)
        self.Bl, self.I_bl = find_unique_blocks_and_indecies(self.Bl,self.I_bl)
        self.Cl, self.I_cl = find_unique_blocks_and_indecies(self.Cl,self.I_cl)
    
    def info(self,diagonal_zeros):
        all_slices=[]
        is_zero=[]
        sx = 0
        for i in range(len(self.I_al)):
            shape_i = self.A(i).shape[0+self.num_vect_inds]
            s = []
            z = []
            sy = 0
            for j in range(len(self.I_al)):
                shape_j=self.A(j).shape[1+self.num_vect_inds]
                s += [[slice(sx,sx+shape_i),slice(sy,sy+shape_j)]]
                sy += shape_j
                if i==j or i==j+1 or i+1==j:
                    if diagonal_zeros==False:
                        z+=[1]
                    elif not all_zero(self.Block(i,j)):
                        z+=[1]
                    else:
                        z+=[0]
                else:
                    z+=[0]
            all_slices+=[s]
            is_zero+=[z]
            sx+=shape_i
        
        self.all_slices=all_slices
        self.is_zero=np.array(is_zero)
        nonzero_slices=[]
        for i in range(len(self.Al)):
            nZ=[]
            for j in range(len(self.Al)):
                if is_zero[i][j]==1:
                    nZ+=[all_slices[i][j]]
            nonzero_slices+=[nZ]
            
        #self.non_zero_slices = nonzero_slices
        self.dtype = self.Al[0].dtype
    
    def _Sequences_XY(self, Print='no', forget_tilde = False):
        self.Xs=[]; self.Xts=[]
        self.Ys=[]; self.Yts=[]
        x = np.zeros(self.A(self.N-1).shape,dtype=self.dt)
        y = np.zeros(self.A(0).shape,dtype=self.dt)
        self.Xs+=[x]
        self.Ys+=[y]
        
        for n in range(0,self.N-1):
            yt = Solve(self.A(n)   -y, self.C(n))
            y  = MM(self.B(n),yt) #self.B(n). dot(yt)
            if forget_tilde == False: self.Yts+=[yt] 
            self.Ys+= [y]
        
        for n in range(self.N-1,0,-1):
            xt = Solve(self.A(n)-x, self.B(n-1))
            x  = MM(self.C(n-1),xt)#self.C(n-1).dot(xt)
            self.Xs +=[x]; 
            if forget_tilde == False: self.Xts+=[xt]
        
        self.Xs=self.Xs[::-1]
        if forget_tilde == False:
            self.Xts=self.Xts[::-1]
            self.Xts=self.Xts+[None]
            self.Yts=[None]+self.Yts
        
        if Print=='si':
            print('X,Y,Yt,Xt calculated')
    
    def Inverse_Diag_of_Diag(self, which_n):
        out = np.zeros(self.shape[:-1], dtype = self.dtype)
        
        iD = []
        for i in which_n:
            iD+=[self.A(i).copy()]
        count_x = self.N
        count_y = -1
        for n in range(-1, self.N-1):
            if n==-1:
                x = np.zeros(self.A(count_x-1).shape,dtype=self.dt)
                y = np.zeros(self.A(count_y+1).shape,dtype=self.dt)
                
            else:
                
                xt = Solve(self.A(count_x  )   - x, self.B(count_x-1))
                x  = MM(   self.C(count_x-1),xt)
                
                yt = Solve(self.A(count_y  )   - y, self.C(count_y  ))
                y  = MM(   self.B(count_y  ),yt)
            
            count_x-=1
            count_y+=1
            if count_x in which_n:
                idx = which_n.index(count_x)
                iD[idx]-=x
            if count_y in which_n:
                idx = which_n.index(count_y)
                iD[idx]-=y
        
        it = 0
        for q in which_n:
            inds_diag = [i for i in range(self.A(q).shape[-1])] 
            Temp = Inv(iD[it])[... , inds_diag, inds_diag]
            out[... , self.all_slices[q][q][0]] = Temp
            it += 1
        
        del xt,x,y,yt,iD
        gc.collect()
        return out
    
    def Gen_Inv_Diag(self,which_n = None,forget_tilde = False):
        self.Clean_inverse()
        self.iMl_dict = {}
        self.iMl_it = 0
        
        if which_n is None:
            which_n = range(self.N)
        
        self._Sequences_XY(forget_tilde = forget_tilde)
        it=int(self.iMl_it)
        for i in which_n:
            self.inds+=[[i,i]]
            self.iMl+=[Inv(self.A(i)-self.X(i)-self.Y(i))]    
            self.iMl_dict.update({(i,i):it})
            it+=1
        self.iMl_it = int(it)
        self.Ys = None
        self.Xs = None
        gc.collect()
    
    def iM(self,i,j):
        try: 
            lind = self.iMl_dict[(i,j)]
            #assert (self.iMl[lind] == self.iM_old(i,j)).all()
            return self.iMl[lind]
        except KeyError:
            print('BTD Inversion failed')
            error()
    
    def Inverse_dot_v(self,v):
        # No explicit inversion, only 1 block at a time
        assert self.shape[-1] == v.shape[0]
        if len(v.shape) == 1:
            shape = self.shape[0:len(self.shape)-1]
        elif len(v.shape) == 2:
            shape = self.shape[0:len(self.shape)-1] + (len(v[0,:]),)
        truth  = v.any(axis = 1)
        n_idx  = []
        tæller = 0
        for s in self.all_slices:
            idx = s[0][0]
            if truth[idx].any():
                n_idx+=[tæller]
            tæller+=1
        
        res = np.zeros(shape,dtype = self.dtype)
        for n in n_idx:
            mtemp = self.iM(n,n)
            res[... , self.all_slices[n][n][0],:]    += MM(mtemp,v[self.all_slices[n][n][1]])
            for m in range(n+1,self.N):
                mtemp = MM(-self.Xt(m-1), mtemp)
                res[... , self.all_slices[m][n][0],:] += MM(mtemp,v[self.all_slices[m][n][1]])
            
            mtemp = self.iM(n,n)
            for m in range(n-1,-1,-1):
                mtemp = MM(-self.Yt(m+1), mtemp)
                res[... , self.all_slices[m][n][0],:] += MM(mtemp,v[self.all_slices[m][n][1]])
        
        #self.Clean_inverse()
        return res
    
    def Invert_from_mask(self, mask):
        assert mask.shape[0] ==  mask.shape[1]
        assert mask.shape[0] ==  self.N
        
        self.Gen_Inv_Diag()
        it = int(self.iMl_it)
        
        for n in range(self.N):
            Mt = self.iM(n,n)
            col = mask[:,n]
            jmin = np.where(col>0)[0].min()
            jmax = np.where(col>0)[0].max()
            for m in range(n+1, jmax+1):
                Mt = MM(-self.Xt(m-1),Mt)
                if col[m]==1:
                    self.iMl+=[Mt]
                    self.inds+=[[m,n]]
                    self.iMl_dict.update({(m,n):it})
                    it+=1
            
            Mt = self.iM(n,n)
            for m in range(n-1,jmin-1,-1):
                Mt = MM(-self.Yt(m+1),Mt)
                if col[m] == 1:
                    self.iMl+=[Mt]
                    self.inds+=[[m,n]]
                    self.iMl_dict.update({(m,n):it})
                    it+=1
        Res = block_sparse(self.inds,self.iMl,self.is_zero.shape,E_grid = self.E_grid)
        self.Clean_inverse()
        return Res
    
    def Invert(self, BW = 'all'):
        if isinstance(BW, np.ndarray):
            return self.Invert_from_mask(BW)
        
        if BW=='all':
            self.Gen_Inv_Diag()
            #Regner alle elementer i inverse matrix
            it = int(self.iMl_it)
            for n in range(self.N):
                for m in range(n+1,self.N):
                    self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                    self.inds+=[[m,n]]
                    self.iMl_dict.update({(m,n):it})
                    it+=1
                    
                for m in range(n-1,-1,-1):
                    self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                    self.inds+=[[m,n]]
                    self.iMl_dict.update({(m,n):it})
                    it+=1
            
            
            Res = block_sparse(self.inds,self.iMl,self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            return Res
        
        elif isinstance(BW,int):
            self.Gen_Inv_Diag()
            #Regner kun elementer et antal steps væk fra diagonalen
            it = int(self.iMl_it)
            
            for n in range(self.N):
                for m in range(n+1,min(self.N,n+1+BW)):
                    self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                    self.inds+=[[m,n]]
                    self.iMl_dict.update({(m,n):it})
                    it+=1
                    
                for m in range(n-1,max(-1,n-1-BW),-1):
                    self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                    self.inds+=[[m,n]]
                    self.iMl_dict.update({(m,n):it})
                    it+=1
                    
            Res = block_sparse(self.inds,self.iMl,self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            return Res
        
        elif BW[0] == 'N':
            if BW=='N': nn=0 
            else:       nn = int(BW[1:])
            
            self.Gen_Inv_Diag()
            #Regner invers matrix elementer i en N form
            it = int(self.iMl_it)
            
            for n in range(self.N):
                if n<=0+nn or n>=self.N-1-nn:
                    for m in range(n+1,self.N):
                        self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                        self.inds+=[[m,n]]
                        self.iMl_dict.update({(m,n):it})
                        it+=1
                    
                    for m in range(n-1,-1,-1):
                        self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                        self.inds+=[[m,n]]
                        self.iMl_dict.update({(m,n):it})
                        it+=1
                    
            Res = block_sparse(self.inds,self.iMl,self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            gc.collect()
            return Res
        
        elif BW[0]=='Z':
            if BW=='Z': nn=0 
            else:       nn = int(BW[1:])
            
            self.do_transposition()
            self.Gen_Inv_Diag()
            
            it = int(self.iMl_it)
            
            for n in range(self.N):
                if n<=0+nn or n>=self.N-1-nn:
                    for m in range(n+1,self.N):
                        self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                        self.inds+=[[m,n]]
                        self.iMl_dict.update({(m,n):it})
                        it+=1
                    
                    for m in range(n-1,-1,-1):
                        self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                        self.inds+=[[m,n]]
                        self.iMl_dict.update({(m,n):it})
                        it+=1
            
            self.do_transposition()
            Res = block_sparse(self.inds,self.iMl,self.is_zero.shape,E_grid = self.E_grid)
            Res.do_transposition()
            self.Clean_inverse()
            gc.collect()
            return Res
        elif BW == 'Upper':
            self.Gen_Inv_Diag()
            #Regner alle elementer i inverse matrix
            it = int(self.iMl_it)
            for n in range(self.N):
                for m in range(n+1,self.N):
                    self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                    self.inds+=[[m,n]]
                    self.iMl_dict.update({(m,n):it})
                    it+=1
                
            Res = block_sparse(self.inds,self.iMl,self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            return Res
        
        
        elif BW[0:3]=='*\*':
            if BW=='*\*':nn=0
            else: nn=int(BW[3:])
            
            diag_ele = [0]
            for i in range(nn):
                diag_ele+=[i+1]
            diag_ele += [self.N-nn-1]
            for i in range(nn):
                diag_ele+=[self.N - nn + i]
            
            # self.Gen_Inv_Diag()
            self.Gen_Inv_Diag(which_n = diag_ele)
            it = int(self.iMl_it)
            
            for n in range(self.N):
                if n<=0+nn or n>=self.N-1-nn:
                    for m in range(n+1,self.N):
                        if m <= nn or n>=self.N-nn-1:
                            # print('diag loop1: ', m,n)
                            self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                            self.inds+=[[m,n]]
                            self.iMl_dict.update({(m,n):it})
                            it+=1
            
                    for m in range(n-1,-1,-1):
                        if n <= nn or m >= self.N-nn-1:
                            # print('diag loop2: ',m,n)
                            self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                            self.inds+=[[m,n]]
                            self.iMl_dict.update({(m,n):it})
                            it+=1
            # print(self.inds)
            for n in range(nn+1):
                mt = nn
                # print('offdiag loop 1: ',mt,n)
                M_temp = self.iM(mt, n)
                #Propagate through region we dont need
                for j in range(self.N-2*(nn+1)):
                    M_temp = MM(-self.Xt(mt), M_temp)
                    mt+=1
                
                #Save the matrix elements we need
                for j in range(nn+1):
                    M_temp    = MM(-self.Xt(mt), M_temp)
                    self.iMl += [M_temp]
                    mt+=1
                    self.inds+= [[mt,n]]
                    self.iMl_dict.update({(mt,n): it})
                    it+=1
                
            for n in range(self.N - (nn + 1), self.N):
                mt = self.N - (nn + 1)
                # print('offdiag loop 2: ',mt,n)
                M_temp = self.iM(mt, n)
                
                for j in range(self.N - 2*(nn+1)):
                    M_temp = MM(- self.Yt(mt), M_temp)
                    mt-=1
                
                for j in range(nn+1):
                    M_temp = MM(-self.Yt(mt), M_temp)
                    self.iMl += [M_temp]
                    mt-=1
                    self.inds+=[[mt,n]]
                    self.iMl_dict.update({(mt,n): it})
                    it+=1
            
                
            
            Res = block_sparse(self.inds,self.iMl,self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            gc.collect()
            return Res
        
        elif BW[0:4]=='diag':
            which_n = BW[4:].split()
            which_n = [int(i) for i in which_n]
            self.Gen_Inv_Diag(which_n = which_n,forget_tilde = True)
            Res = block_sparse(self.inds,self.iMl,self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            gc.collect()
            return Res
    
    
    def Clean_inverse(self):
        self.inds = None
        self.iMl  = None
        self.iMl_dict = None
        self.inds = []
        self.iMl  = []
    
    def Block(self,i,j):
        if i==j:
            return self.A(i)
        if i==j+1:
            return self.B(j)
        if i==j-1:
            return self.C(i)
        else:
            return None
    
    def _A(self,n):
        if n<0 or n>self.N-1:print('Index error A'); error()
        return self.Al[self.I_al[n]]
    
    def _B(self,n):
        if n<0 or n>self.N-2:print('Index error B'); error()
        return self.Bl[self.I_bl[n]]
    
    def _C(self,n):
        if n<0 or n>self.N-2:print('Index error C'); error()
        return self.Cl[self.I_cl[n]]
    
    def A(self, n):
        if self.has_been_transposed == False and self.has_been_conjugated == False:
            return self._A(n)
        if self.has_been_transposed == False and self.has_been_conjugated == True:
            return np.conj(self._A(n))
        if self.has_been_transposed == True  and self.has_been_conjugated == False:
            return Transpose(self._A(n))
        if self.has_been_transposed == True  and self.has_been_conjugated == True:
            return np.conj(Transpose(self._A(n)))
    
    def B(self, n):
        if self.has_been_transposed == False and self.has_been_conjugated == False:
            return self._B(n)
        if self.has_been_transposed == False and self.has_been_conjugated == True:
            return np.conj(self._B(n))
        if self.has_been_transposed == True  and self.has_been_conjugated == False:
            return Transpose(self._C(n))
        if self.has_been_transposed == True  and self.has_been_conjugated == True:
            return np.conj(Transpose(self._C(n)))
    
    def C(self, n):
        if self.has_been_transposed == False and self.has_been_conjugated == False:
            return self._C(n)
        if self.has_been_transposed == False and self.has_been_conjugated == True:
            return np.conj(self._C(n))
        if self.has_been_transposed == True  and self.has_been_conjugated == False:
            return Transpose(self._B(n))
        if self.has_been_transposed == True  and self.has_been_conjugated == True:
            return np.conj(Transpose(self._B(n)))
    
    def X(self,n):
        if n<0 or n>self.N-1:print('Index error X');error()
        return self.Xs[n]
    def Y(self,n):
        if n<0 or n>self.N-1:print('Index error Y');error()
        return self.Ys[n]
    def Yt(self,n):
        if n<1 or n>self.N-1:print('Index error Yt');error()
        return self.Yts[n]
    def Xt(self,n):
        if n<0 or n>self.N-2:print('Index error Xt');error()
        return self.Xts[n]
    
    def Shape(self):
        n0,n1=0,0
        for i in range(len(self.I_al)):
            n0+=self.A(i).shape[0+self.num_vect_inds]
            n1+=self.A(i).shape[1+self.num_vect_inds]
        
        if self.num_vect_inds==0:
            self.shape=(n0,n1)
        elif self.num_vect_inds==1:
            self.shape=(self.A(i).shape[0],n0,n1)
        elif self.num_vect_inds==2:
            self.shape=(self.A(i).shape[0],self.A(i).shape[1],n0,n1)
        elif self.num_vect_inds==3:
            #print('Please visit https://downloadmoreram.com/')
            self.shape=(self.A(i).shape[0],self.A(i).shape[1],self.A(i).shape[2],n0,n1)
        self.Block_shape=(len(self.I_al),len(self.I_al))
    
    def do_transposition(self):
        self.is_zero = self.is_zero.T
        new_all_slices = []
        for j in range(self.Block_shape[1]):
            new_all_slices_j = []
            for i in range(self.Block_shape[0]):
                new_all_slices_j+=[self.all_slices[i][j][::-1]]
            new_all_slices+=[new_all_slices_j]
        
        self.all_slices = new_all_slices
        self.has_been_transposed = not self.has_been_transposed
    
    def do_conjugation(self):
        self.has_been_conjugated = not self.has_been_conjugated
    
    
    
    def do_dag(self):
        self.do_transposition()
        self.do_conjugation()
    
    def copy(self):
        new_Al = [self.Al[i].copy() for i in range(len(self.Al))]
        new_Bl = [self.Bl[i].copy() for i in range(len(self.Bl))]
        new_Cl = [self.Cl[i].copy() for i in range(len(self.Cl))]
        if self.E_grid is not None:
            E_grid_new = self.E_grid.copy()
        else:
            E_grid_new = None
        return block_td(new_Al,new_Bl,new_Cl,self.I_al.copy(),self.I_bl.copy(),self.I_cl.copy(),self.diagonal_zeros.copy(),E_grid = E_grid_new)
    
    #Kopieret fra block_sparse
    def Tr(self,Ei = None):
        if Ei is None:
            return block_TRACE(self)
        else:
            return block_TRACE_interpolated(self,Ei)
    
    def TrProd(self,A,Ei1=None,Ei2=None,warning='yes'):
        if Ei1 is None and Ei2 is None:
            return block_TRACEPROD(self,A)
        else:
            return block_TRACEPROD_interpolated(self,A,Ei1,Ei2)
    
    def SumAll(self):
        return block_SUMALL(self)
    
    def SumAllMatrixEntries(self):
        return block_SUMALLMATRIXINDECIES(self)
    
    def BDot(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_MATMUL(self,A)
        else:
            return block_MATMUL_interpolated(self,A,Ei1,Ei2)
    
    def Add(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_ADD(self,A)
        else:
            return block_ADD_interpolated(self,A,Ei1,Ei2)
    
    def Subtract(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_SUBTRACT(self,A)
        else:
            return block_SUBTRACT_interpolated(self,A,Ei1,Ei2)
    
    def MulEleWise(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_MULELEMENTWISE(self,A)
        else:
            return block_MULELEMENTWISE_interpolated(self,A,Ei1,Ei2)
    
    def A_Adag(self,Ei1,Ei2):
        return block_A_Adag_Interpolated(self,Ei1,Ei2)
    
    def scalar_add(self, s):
        return block_td([al + s for al in self.Al],
                        [bl + s for bl in self.Bl],
                        [cl + s for cl in self.Cl],
                        self.I_al.copy(),
                        self.I_bl.copy(),
                        self.I_cl.copy(),
                        diagonal_zeros = self.diagonal_zeros, E_grid = self.E_grid)
                        
    def scalar_multiply(self, s):
        return block_td([al * s for al in self.Al],
                        [bl * s for bl in self.Bl],
                        [cl * s for cl in self.Cl],
                        self.I_al.copy(),
                        self.I_bl.copy(),
                        self.I_cl.copy(),
                        diagonal_zeros = self.diagonal_zeros, E_grid = self.E_grid)



class block_sparse:
    def __init__(self, inds, vals,Block_shape,E_grid=None,FoRcE_dTypE = None):
        self.inds = inds.copy()
        self.vals = [v.copy() for v in vals]
        
        self.Block_shape = Block_shape
        self.FoRcE_dTypE = FoRcE_dTypE
        
        self.has_been_conjugated = False
        self.has_been_transposed = False
        
        self.info()
        
        if np.any(self.is_zero):
            self.num_vect_inds = max([len(vals[i].shape)-2 for i in range(len(vals))])
        else:
            self.num_vec_inds = 0
        
        self.E_grid = E_grid
    
    # def _Block_old_dont_use(self,i,j):
    #     lind=inds_to_lind([i,j],self.inds)
    #     if lind is not None:
    #         return self.vals[lind]
    #     else: 
    #         return None
    
    
    def Block(self,i,j):
        
        if hasattr(self, 'Symmetric'):
            try:
                try:
                    lind = self.ind_dict[(i,j)]
                    return self.vals[lind]
                except:
                    pass
                try:
                    lind = self.ind_dict[(j,i)]
                    return Transpose(self.vals[lind])
                except:
                    pass
                    
            except KeyError:
                return None
        
        if self.has_been_transposed == False and self.has_been_conjugated == False:
            try:
                lind = self.ind_dict[(i,j)]
                return self.vals[lind]
            except KeyError:
                return None
        if self.has_been_transposed == False and self.has_been_conjugated == True:
            try:
                lind = self.ind_dict[(i,j)]
                return np.conj(self.vals[lind])
            except KeyError:
                return None
        if self.has_been_transposed == True and self.has_been_conjugated == False:
            try:
                lind = self.ind_dict[(j,i)]
                return Transpose(self.vals[lind])
            except KeyError:
                return None
        if self.has_been_transposed == True and self.has_been_conjugated == True:
            try:
                lind = self.ind_dict[(j,i)]
                return np.conj(Transpose(self.vals[lind]))
            except KeyError:
                return None
    
    def info(self):
        inds_d = {}
        it = 0
        for ind in self.inds:
            inds_d.update({(ind[0],ind[1]):it})
            it+=1
        self.ind_dict = inds_d
        it=0
        
        is_zero=[]
        n0=self.Block_shape[0]
        n1=self.Block_shape[1]
        
        for i in range(n0):
            z = []
            for j in range(n1):
                if not all_zero(self.Block(i,j)):
                    z+=[1]
                else:
                    z+=[0]
            
            is_zero+=[z]
        
        self.is_zero=np.array(is_zero)
        
        if self.FoRcE_dTypE is None:
            self.dtype = self.vals[0].dtype
        else:
            self.dtype = self.FoRcE_dTypE
    
    
    
    def do_transposition(self):
        self.has_been_transposed = not self.has_been_transposed
        self.is_zero = self.is_zero.T
        self.Block_shape = self.Block_shape[::-1]
    
    def do_conjugation(self):
        self.has_been_conjugated =  not self.has_been_conjugated
    
    
    
    
    
    def do_dag(self):
        self.do_transposition()
        self.do_conjugation()
    
    def copy(self):
        vals_new = [self.vals[i].copy() for i in range(len(self.vals))]
        inds_new = self.inds.copy()
        if self.E_grid is not None:
            E_grid_new = self.E_grid.copy()
        else:
            E_grid_new = None
        BS = self.Block_shape.copy()
        if self.has_been_transposed:
            BS = BS[::-1]
        return block_sparse(inds_new, vals_new, BS, E_grid = E_grid_new)
    
    def Tr(self,Ei = None):
        if Ei is None:
            return block_TRACE(self)
        else:
            return block_TRACE_interpolated(self,Ei)
    
    def TrProd(self,A,Ei1=None,Ei2=None,warning='yes'):
        if Ei1 is None and Ei2 is None:
            return block_TRACEPROD(self,A)
        else:
            return block_TRACEPROD_interpolated(self,A,Ei1,Ei2)
    
    def SumAll(self):
        return block_SUMALL(self)
    
    def SumAllMatrixEntries(self):
        return block_SUMALLMATRIXINDECIES(self)
    
    def BDot(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_MATMUL(self,A)
        else:
            return block_MATMUL_interpolated(self,A,Ei1,Ei2)
    
    def Add(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_ADD(self,A)
        else:
            return block_ADD_interpolated(self,A,Ei1,Ei2)
    def Subtract(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_SUBTRACT(self,A)
        else:
            return block_SUBTRACT_interpolated(self,A,Ei1,Ei2)
    def MulEleWise(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_MULELEMENTWISE(self,A)
        else:
            return block_MULELEMENTWISE_interpolated(self,A,Ei1,Ei2)
    
    def Make_BTD(self, Mask = None) :
        Al = []; Bl = []; Cl = []
        Ia = []; Ib = []; Ic = []
        if self.Block_shape[0] !=  self.Block_shape[1]:
            print('Matrix is not square, so cant convert block trididagonal matrix')
            return
        bs = self.Block_shape[0]
        if Mask is None:
            Mask = [[], [], []]
            i_idx = np.arange(0,bs,2)
            if np.mod(bs,2)==1:
                i_idx = i_idx[0:len(i_idx)-1]
            for i in i_idx:
                
                Mask[1] += [np.array([
                               [
                                    [i,i],  [i,i+1]
                               ]
                           ,
                               [
                                    [i+1,i],[i+1,i+1]    
                               ]
                                    ])
                           ]
                if i != i_idx[-1]:
                    Mask[0] += [np.array([
                                    [
                                        [i+2,i], [i+2,i+1]
                                    ]
                                    ,
                                    [
                                        [i+3,i], [i+3,i+1]
                                    ]
                                        ])
                               ]
                    Mask[2] += [np.array([
                                    [
                                        [i,i+2],   [i,  i+3]
                                    ]
                               ,
                                    [
                                        [i+1,i+2], [i+1,i+3]
                                    ]
                                        ])
                                ]
            
            if np.mod(bs,2) == 1:
                Mask[1] += [np.array([
                            [
                                [i+2,i+2]
                            ]
                                      ])
                            ]
                Mask[0] += [np.array([
                            [
                                [i+2,i  ], [i+2,i+1]
                            ]
                                    ])
                            ]
                Mask[2] += [np.array([
                            [
                                [i,  i+2]
                            ]       
                            ,
                            [
                                [i+1,i+2]
                            ]
                                    ])
                            ]
        Shapes = [[], [], []]
        pcount = 0
        for P in Mask:
            never_use_i= 0
            for idx in P:
                s = idx.shape
                ske = []
                ske_shape = []
                for i in range(s[0]):
                    let = []
                    let_shape  = []
                    for j in range(s[1]):
                        bi = idx[i,j,0]
                        bj = idx[i,j,1]
                        
                        if self.Block(bi,bj) is not None:
                            BB = self.Block(bi,bj).copy()
                        else:
                            for k in range(self.Block_shape[0]):
                                tb = self.Block(k, bj)
                                if tb is not None:
                                    col_s = tb.shape[-1]
                                    break
                                elif k == self.Block_shape[0]-1:
                                    print('error in making BTD????')
                                    assert 1 == 2
                            for k in range(self.Block_shape[1]):
                                tb = self.Block(bi, k)
                                if tb is not None:
                                    row_s = tb.shape[-2]
                                    break
                                elif k == self.Block_shape[1]-1:
                                    print('error in making BTD????')
                                    assert 1 == 2
                            
                            zero_shape = self.vals[0][...,0,0].shape + (row_s,col_s)
                            BB = np.zeros(zero_shape, dtype = self.dtype).copy()
                        let+=[BB]
                        let_shape+=[BB.shape]
                    ske+=[let]
                    ske_shape += [let_shape]
                
                Block = np.block(ske)
                if pcount == 1:
                    Al+=[Block]; Ia+=[never_use_i]
                if pcount == 0:
                    Bl+=[Block]; Ib+=[never_use_i]
                if pcount == 2:
                    Cl+=[Block]; Ic+=[never_use_i]
                Shapes[pcount] += [ske_shape]
                
                never_use_i+=1
            pcount+=1
        
        Out = block_td(Al,Bl,Cl,Ia,Ib,Ic,diagonal_zeros = False, E_grid = self.E_grid)
        Out.BTD_Mask = Mask
        Out.shapes_mask = Shapes
        
        return Out
    
    def Get_diagonal(self, slices):
        n = min(self.Block_shape)
        num = slices[-1][-1][0].stop
        s = self.vals[0][...,0,0].shape+(num,)
        diag = np.zeros(s, dtype = self.dtype)
        
        for i in range(n):
            si = slices[i][i][0]
            block_ii = self.Block(i,i)
            if block_ii is not None:
                vals = []
                
                for i in range(si.stop-si.start):
                     diag[..., si.start + i] =  block_ii[ ... ,i,i]
        
        return diag
    
    def get_e_subset(self, idx):
        #inds, vals,Block_shape,E_grid
        return block_sparse(self.inds, 
                            [v[... , idx, :,:] for v in self. vals], 
                            Block_shape = self.Block_shape,
                            E_grid = None)
    
   
    
    def tonp(self,slices):
        return Blocksparse2Numpy(self,slices)
    
    def eig(self,slices, hermitian = False, as_dense = False):
        if hermitian:
            EIG = np.linalg.eigh
        else:
            EIG = np.linalg.eig
        
        print('\n Diagonalising matrix!\n')
        # meant for taking squareroots of scattering matrices
        # lazily implemented using scipy.linalg.sqrtm
        
        nv = self.num_vect_inds
        idx,jdx = np.where(self.is_zero != 0)
        
        i_min, i_max = idx.min(), idx.max()
        j_min, j_max = jdx.min(), jdx.max()
        
        mat =  []
        def len_slice(s):
            return s.stop - s.start
        
        S = self.vals[0].shape[:-2]
        subslices = []
        
        for i in range(i_min, i_max+1):
            lines = []
            subsubslices = []
            for j in range(j_min, j_max+1):
                si, sj = slices[i][j]
                B = self.Block(i,j)
                if B is None:
                    B = np.zeros(S + (len_slice(si), len_slice(sj)), dtype = self.dtype)
                lines += [B]
                subsubslices+=[[si, sj]]
            mat += [lines]
            subslices+=[subsubslices]
        
        sub_mat = np.block(mat)
        e,v  = EIG(sub_mat)
        if as_dense:
            return e, v, slices[i_min][i_min][0].start, slices[i_max][i_max][0].stop-1
        
        
        def zero_slices(inp_slices):
            new_slices = []
            zero =  inp_slices[0][0][0].start,inp_slices[0][0][1].start
            for rows in inp_slices:
                new_cols = []
                for cols in rows:
                    si, sj = cols
                    sin = slice(si.start - zero[0], si.stop - zero[0]) 
                    sjn = slice(sj.start - zero[1], sj.stop - zero[1])
                    new_cols += [ [sin, sjn] ]
                new_slices += [ new_cols ]
            return new_slices
        
        zeroed_slices = zero_slices(subslices)
        new_idx            = []
        new_blocks         = []
        new_idx_diag       = []
        new_blocks_diag    = []
        
        for i in range(i_max - i_min +1):
            for j in range(j_max  - j_min +1):
                new_idx    += [[i + i_min,j + j_min]]
                si,sj = zeroed_slices[i][j]
                new_blocks += [v[..., si,sj]]
                if i == j:
                    new_idx_diag   += [[i + i_min,j + j_min]]
                    nvals = si.stop - si.start 
                    m = np.zeros(S + (nvals , nvals), dtype = e.dtype)
                    midx  = np.arange(si.stop - si.start)
                    m[..., midx,midx] = e[..., si]
                    new_blocks_diag+= [m.copy()]
        return block_sparse(new_idx_diag, new_blocks_diag, self.Block_shape, E_grid = self.E_grid), block_sparse(new_idx     , new_blocks     , self.Block_shape, E_grid = self.E_grid)
        
    def sqrt(self, slices):
        print('using scipy.linalg.sqrtm + for loops, pretty slow')
        # meant for taking squareroots of scattering matrices
        # lazily implemented using scipy.linalg.sqrtm
        
        nv = self.num_vect_inds
        idx,jdx = np.where(self.is_zero != 0)
        
        i_min, i_max = idx.min(), idx.max()
        j_min, j_max = jdx.min(), jdx.max()
        
        mat =  []
        def len_slice(s):
            return s.stop - s.start
        
        S = self.vals[0].shape[:-2]
        subslices = []
        
        for i in range(i_min, i_max+1):
            lines = []
            subsubslices = []
            for j in range(j_min, j_max+1):
                si, sj = slices[i][j]
                B = self.Block(i,j)
                if B is None:
                    B = np.zeros(S + (len_slice(si), len_slice(sj)), dtype = self.dtype)
                lines += [B]
                subsubslices+=[[si, sj]]
            mat += [lines]
            subslices+=[subsubslices]
        
        sub_mat = np.block(mat)
        sqrt    = sqrtm_on_KLMij(sub_mat)
        
        def zero_slices(inp_slices):
            new_slices = []
            zero =  inp_slices[0][0][0].start,inp_slices[0][0][1].start
            for rows in inp_slices:
                new_cols = []
                for cols in rows:
                    si, sj = cols
                    sin = slice(si.start - zero[0], si.stop - zero[0]) 
                    sjn = slice(sj.start - zero[1], sj.stop - zero[1])
                    new_cols += [ [sin, sjn] ]
                new_slices += [ new_cols ]
            return new_slices
        
        zeroed_slices = zero_slices(subslices)
        new_idx       = []
        new_blocks    = []
        for i in range(i_max - i_min +1):
            for j in range(j_max  - j_min +1):
                new_idx    += [[i + i_min,j + j_min]]
                si,sj = zeroed_slices[i][j]
                new_blocks += [sqrt[..., si,sj]]
        
        return block_sparse(new_idx, new_blocks, self.Block_shape, E_grid = self.E_grid)
    
    def scalar_add(self, s):
        return block_sparse(self.inds.copy(), 
                            [v + s for v in self.vals],
                            self.Block_shape, E_grid = self.E_grid)
    def scalar_multiply(self, s):
        return block_sparse(self.inds.copy(), 
                            [v * s for v in self.vals],
                            self.Block_shape, E_grid = self.E_grid)

def block_TRACE(A):
    n=A.is_zero.shape[0]
    return sum([np.trace(A.Block(i,i),axis1=-1,axis2=-2) for i in range(n) if A.Block(i,i) is not None])

def block_SUMALLMATRIXINDECIES(A):
    I,J = Wh(A.is_zero>0)
    n_ind = len(I)
    if n_ind==0:
        return 0
    else:
        S=[]
        for tæl in range(n_ind):
            i,j = I[tæl],J[tæl]
            S+=[np.sum(A.Block(i,j),axis=(-1,-2))]
    return sum(S)

def block_SUMALL(A):
    return np.sum(block_SUMALLMATRIXINDECIES(A))

def block_TRACEPROD(A1,A2):
    Res = block_SUMALLMATRIXINDECIES(block_MULELEMENTWISE_TRANSPOSE_LAST(A1, A2))
    return Res

def block_MATMUL(A1,A2):
    assert A1.Block_shape[1]==A2.Block_shape[0]
    Prod_pat = A1.is_zero.dot(A2.is_zero)
    I,J =  Wh(Prod_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        k1 = Wh(A1.is_zero[i,:]==1)[0]
        k2 = Wh(A2.is_zero[:,j]==1)[0]
        K=np.intersect1d(k1,k2)
        if len(K)>0:
            Res_inds+=[[i,j]]
            # First = MM(A1.Block(i,K),A2.Block(K,j)).sum(axis = 0)
            First = MM(A1.Block(i,K[0]),A2.Block(K[0],j))
            for k in K[1:]:
                First       += MM(A1.Block(i,k),A2.Block(k,j))
            Res_vals += [First]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A2.is_zero.shape[1]))


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def block_ADD(A1,A2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    #assert A1.num_vect_inds==A2.num_vect_inds
    
    Sum_pat = A1.is_zero + A2.is_zero
    I,J =  Wh(Sum_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        if A1.Block(i,j) is not None and A2.Block(i,j) is not None:
            Res_vals += [A1.Block(i,j)+A2.Block(i,j)]
        if A1.Block(i,j) is not None and A2.Block(i,j)     is None:
            Res_vals += [A1.Block(i,j)              ]
        if A1.Block(i,j) is None and A2.Block(i,j)     is not None:
            Res_vals += [A2.Block(i,j)              ]
            
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A1.is_zero.shape[1]))

def block_SUBTRACT(A1,A2):
    assert A1.Block_shape==A2.Block_shape
    
    Sum_pat = A1.is_zero + A2.is_zero
    I,J =  Wh(Sum_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        if A1.Block(i,j) is not None and A2.Block(i,j) is not None:
            Res_vals += [ A1.Block(i,j) - A2.Block(i,j)]
        if A1.Block(i,j) is not None and A2.Block(i,j)     is None:
            Res_vals += [ A1.Block(i,j)              ]
        if A1.Block(i,j) is None and A2.Block(i,j)     is not None:
            Res_vals += [-A2.Block(i,j)              ]
    
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A1.is_zero.shape[1]))

def block_MULELEMENTWISE(A1,A2):
    assert A1.Block_shape==A2.Block_shape
    Mul_pat = A1.is_zero * A2.is_zero
    I,J =  Wh(Mul_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        Res_vals+=[A1.Block(i,j)*A2.Block(i,j)]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),A1.is_zero.shape)

def block_MULELEMENTWISE_TRANSPOSE_LAST(A1,A2):
    assert A1.Block_shape==A2.Block_shape
    Mul_pat = A1.is_zero * A2.is_zero.T
    I,J =  Wh(Mul_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        Res_vals+=[A1.Block(i,j)*Transpose(A2.Block(j,i))]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),A1.is_zero.shape)


def block_TRACE_interpolated(A,Ei):
    assert not isinstance(A.E_grid,type(None))
    F,IND = Interpolate(A.E_grid,Ei)
    n=A.is_zero.shape[0]
    return sum([np.trace(Interpolate_block(A.Block(i,i),F,IND),axis1=-1,axis2=-2) for i in range(n) if A.Block(i,i) is not None])

def block_SUMALLMATRIXINDECIES_interpolated(A,Ei):
    assert not isinstance(A.E_grid,type(None))
    F,IND = Interpolate(A.E_grid,Ei)
    I,J = Wh(A.is_zero>0)
    n_ind = len(I)
    if n_ind==0:
        return 0
    else:
        S=[]
        for tæl in range(n_ind):
            i,j = I[tæl],J[tæl]
            S+=[np.sum(Interpolate_block(A.Block(i,j),F,IND),axis=(-1,-2))]
    return sum(S)

def block_SUMALL_interpolated(A,Ei):
    return np.sum(block_SUMALLMATRIXINDECIES_interpolated(A,Ei))

def block_MULELEMENTWISE_interpolated(A1,A2,Ei1,Ei2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    #assert A1.num_vect_inds==A2.num_vect_inds
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    F1,IND1 = Interpolate(A1.E_grid,Ei1)
    F2,IND2 = Interpolate(A2.E_grid,Ei2)
    
    Mul_pat = A1.is_zero * A2.is_zero
    I,J =  Wh(Mul_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        
        Res_vals+=[Interpolate_block(A1.Block(i,j),F1,IND1)*Interpolate_block(A2.Block(i,j),F2,IND2)]
        
    return block_sparse(Res_inds.copy(),Res_vals.copy(),A1.is_zero.shape)

def block_MULELEMENTWISE_TRANSPOSE_LAST_interpolated(A1,A2,Ei1,Ei2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    # assert A1.num_vect_inds==A2.num_vect_inds
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    
    F1,IND1 = Interpolate(A1.E_grid,Ei1)
    F2,IND2 = Interpolate(A2.E_grid,Ei2)
    
    Mul_pat = A1.is_zero * A2.is_zero.T
    I,J =  Wh(Mul_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        Res_vals+=[Interpolate_block(A1.Block(i,j),F1,IND1)*Transpose(Interpolate_block(A2.Block(j,i),F2,IND2))]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),A1.is_zero.shape)

def block_TRACEPROD_interpolated(A1,A2,Ei1,Ei2,warning='yes'):
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    Res = block_SUMALLMATRIXINDECIES(block_MULELEMENTWISE_TRANSPOSE_LAST_interpolated(A1, A2, Ei1, Ei2))
    return Res

def block_MATMUL_interpolated(A1,A2,Ei1,Ei2):
    assert A1.dtype==A2.dtype
    assert A1.Block_shape[1]==A2.Block_shape[0]
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    
    F1,IND1  = Interpolate(A1.E_grid,Ei1)
    F2,IND2  = Interpolate(A2.E_grid,Ei2)
    
    Prod_pat = A1.is_zero.dot(A2.is_zero)
    I,J =  Wh(Prod_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        k1 = Wh(A1.is_zero[i,:]==1)[0]
        k2 = Wh(A2.is_zero[:,j]==1)[0]
        K=np.intersect1d(k1,k2)
        if len(K)>0:
            Res_inds+=[[i,j]]
            First = MM(Interpolate_block(A1.Block(i,K[0]),F1,IND1),Interpolate_block(A2.Block(K[0],j),F2,IND2))
            for k in K[1:]:
                First += MM(Interpolate_block(A1.Block(i,k),F1,IND1),Interpolate_block(A2.Block(k,j),F2,IND2))
            Res_vals += [First]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A2.is_zero.shape[1]))

def block_A_Adag_Interpolated(A,Ei1,Ei2):
    assert not isinstance(A.E_grid,type(None))
    
    F1,IND1  = Interpolate(A.E_grid,Ei1)
    F2,IND2  = Interpolate(A.E_grid,Ei2)
    
    Prod_pat = A.is_zero.dot(A.is_zero)
    I,J =  Wh(Prod_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        k1 = Wh(A.is_zero  [i,:]==1)[0]
        k2 = Wh(A.is_zero.T[:,j]==1)[0]
        K=np.intersect1d(k1,k2)
        if len(K)>0:
            Res_inds+=[[i,j]]
            First = MM(
                       Interpolate_block(A.Block(i,K[0]),F1,IND1),
                       Interpolate_block(
                                         Transpose(A.Block(j,K[0])).conj(),F2,IND2
                                        )
                       )
            
            for k in K[1:]:
                First += MM(
                            Interpolate_block(A.Block(i,k),F1,IND1),
                            Interpolate_block(
                                              Transpose(A.Block(j,k)).conj(),F2,IND2
                                             )
                           )
            
            Res_vals += [First]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A.is_zero.shape[0],A.is_zero.shape[0]))

def block_ADD_interpolated(A1,A2,Ei1,Ei2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    # assert A1.num_vect_inds==A2.num_vect_inds
    
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    
    F1,IND1 = Interpolate(A1.E_grid,Ei1)
    F2,IND2 = Interpolate(A2.E_grid,Ei2)
    
    Sum_pat = A1.is_zero + A2.is_zero
    I,J =  Wh(Sum_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        if A1.Block(i,j) is not None and A2.Block(i,j) is not None:
            Res_vals += [Interpolate_block(A1.Block(i,j),F1,IND1) + Interpolate_block(A2.Block(i,j),F2,IND2)]
        if A1.Block(i,j) is not None and A2.Block(i,j)     is None:
            Res_vals += [Interpolate_block(A1.Block(i,j),F1,IND1)              ]
        if A1.Block(i,j) is None and A2.Block(i,j)     is not None:
            Res_vals += [Interpolate_block(A2.Block(i,j),F2,IND2)              ]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A1.is_zero.shape[1]))

def block_SUBTRACT_interpolated(A1,A2,Ei1,Ei2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    # assert A1.num_vect_inds==A2.num_vect_inds
    
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    
    F1,IND1 = Interpolate(A1.E_grid,Ei1)
    F2,IND2 = Interpolate(A2.E_grid,Ei2)
    
    Sum_pat = A1.is_zero + A2.is_zero
    I,J =  Wh(Sum_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        if A1.Block(i,j) is not None and A2.Block(i,j) is not None:
            Res_vals += [ Interpolate_block(A1.Block(i,j),F1,IND1)-Interpolate_block(A2.Block(i,j),F2,IND2)]
        if A1.Block(i,j) is not None and A2.Block(i,j)     is None:
            Res_vals += [ Interpolate_block(A1.Block(i,j),F1,IND1)              ]
        if A1.Block(i,j) is None and A2.Block(i,j)     is not None:
            Res_vals += [-Interpolate_block(A2.Block(i,j),F2,IND2)              ]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A1.is_zero.shape[1]))

def block_TRACE_different_bs(A,a,S,s):
    #Matrix with more grainy block structure goes on the right....
    if 'block_td' in  A.__str__():
        Res = np.zeros( A.Al[0][..., 0, 0].shape, dtype = A.dtype )
    else:
        Res = np.zeros( A.vals[0][..., 0, 0].shape, dtype = A.dtype )
    assert S[-1][-1][0].stop == s[-1][-1][0].stop
    assert S[-1][-1][1].stop == s[-1][-1][1].stop
    
    idx_A = np.where(A.is_zero==1)
    idx_a = np.where(a.is_zero==1)
    used_idx = []
    for tæller in range(len(idx_A[0])):
        I,J = idx_A[0][tæller], idx_A[1][tæller]
        sA0, sA1 = S[I][J][0], S[I][J][1]
        z0,  z1 = sA0.start, sA1.start
        sA0, sA1= slice(0, sA0.stop - z0), slice(0, sA1.stop - z1)
        B = A.Block(I,J)
        for rellæt in range(len(idx_a[0])):
            #Tr AB = sum(A .* B.T)
            j,i = idx_a[0][rellæt], idx_a[1][rellæt]
            sa0, sa1 = s[i][j][0], s[i][j][1]
            sa0, sa1 = slice( sa1.start - z0, sa1.stop - z0), slice( sa0.start - z1, sa0.stop - z1)
            b    = Transpose(a.Block(i,j))
            if sA0.start<= sa0.start < sA0.stop and sA0.start<= sa0.stop <= sA0.stop:
                if sA1.start<= sa1.start < sA1.stop and sA1.start<= sa1.stop <= sA1.stop:
                    # print(sa0,sa1, sA0, sA1)
                    if [j,i] not in used_idx:
                        Res += np.sum(B[..., sa0,sa1] * b, axis = (-1,-2) )
                        used_idx += [[j,i]]
    
    return Res

@jit
def Interp(E0,E):
    assert E.min()>=E0.min()
    assert E.max()< E0.max()
    F = np.zeros((len(E),2))
    inds = np.zeros((len(E),2))
    ne=len(E)
    ne0=len(E0)
    for j in range(ne):
        e=E[j].real
        for i in range(ne0-1):
            if E0[i]<=e<E0[i+1]:
                dE     = E0[i+1]-E0[i]
                de     = e-E0[i]
                f = de/dE
                F[j,0] = 1-f
                F[j,1] = f
                inds[j,0] = i
                inds[j,1] = i+1
                break
    
    return F,inds

def Interpolate(E0,E):
    f,i = Interp(E0,E)
    return f, i.astype(int)

def Interpolate_block(Arr,f,i):
    #Always interpolates in the third index
    if len(Arr.shape)==4:
        return (Arr[:,i[:,0],:,:].transpose(0,2,3,1)*f[:,0]).transpose(0,3,1,2) + (Arr[:,i[:,1],:,:].transpose(0,2,3,1)*f[:,1]).transpose(0,3,1,2)
    elif len(Arr.shape)==2:
        return Arr
    else:
        print('Interpolate_block give array that it isnt compatible with\n')
        assert 1==0

def Compare_nd_and_btd(NP, B):
    Bs = B.Block_shape
    T=np.zeros(Bs)
    for i in range(Bs[0]):
        for j in range(Bs[1]):
            
            if len(NP.shape) == 4:
                denseblock = NP[:,:,B.all_slices[i][j][0],B.all_slices[i][j][1]]
            elif len(NP.shape)== 3:
                denseblock = NP[:,B.all_slices[i][j][0],B.all_slices[i][j][1]]
            elif len(NP.shape)== 2:
                denseblock = NP[B.all_slices[i][j][0],B.all_slices[i][j][1]]
            
            if B.Block(i,j) is not None:
                T[i,j]  = np.isclose(denseblock,B.Block(i,j)).all()
            elif B.Block(i,j) is None and (denseblock==0).all():
                T[i,j] = 1
    return T

def Blocksparse2Numpy(A,slices,iti_start = 0, itj_start = 0):
    ni = slices[-1][-1][0].stop 
    nj = slices[-1][-1][1].stop
    if iti_start != 0 or itj_start != 0:
        ni-=slices[0][0][0].start
        nj-=slices[0][0][1].start
        
    bs = A.Block_shape
    breaki = False
    for i in range(bs[0]):
        for j in range(bs[1]):
            if A.Block(i,j) is not None:
                shape = A.Block(i,j).shape
                breaki = True
                break
        if breaki == True:
            break
    ls = len(shape)
    if ls == 2:
        s  = (ni,nj)
    elif ls == 3:
        s  = (shape[0],ni,nj)
    elif ls == 4:
        s  = (shape[0],shape[1],ni,nj)
    elif ls == 5:
        s  = (shape[0],shape[1],shape[2],ni,nj)
    
    Full = np.zeros(s,A.dtype)
    iti = iti_start
    counter_i = slices[0][0][0].start
    counter_j = slices[0][0][1].start
    
    for Si in slices:
        itj = itj_start
        
        for Sj in Si:
            si = Sj[0]; sj = Sj[1]
            si_new = slice(si.start-counter_i,si.stop-counter_i)
            sj_new = slice(sj.start-counter_j,sj.stop-counter_j)
            if A.Block(iti,itj) is not None:
                if ls == 2: 
                    Full[si_new,sj_new] = A.Block(iti,itj)
                elif ls == 3: 
                    Full[:,si_new,sj_new] = A.Block(iti,itj)
                elif ls == 4: 
                    Full[:,:,si_new,sj_new] = A.Block(iti,itj)
                elif ls == 5: 
                    Full[:,:,:,si_new,sj_new] = A.Block(iti,itj)
            itj+=1
        iti+=1
        
    return Full

# def todense(block_matrix, slices, idx_i, idx_j):
#     shape = block_matrix.vals[0].shape[:-2]
#     shape = shape + (len(idx_i), len(idx_j))
#     Out = np.zeros(shape, dtype = block_matrix.dtype)
#     def glob_to_loc(I,J):
#         for i, s in enumerate(slices):
#             for j, ss in enumerate(s):
#                 si, sj = ss
#                 i_start, i_stop = si.start, si.stop
#                 j_start, j_stop = sj.start, sj.stop
#                 if i_start<=I<i_stop and j_start<=J<j_stop:
#                     return (i,j), (I-i_start, J-j_start)
#         return 'error'
    
#     where_i = [ glob_to_loc(i,0) for i in idx_i ]
#     where_j = [ glob_to_loc(0,j) for j in idx_j ]
    
#     Blocks  = [] 
#     loc_idx = [] 
#     sub_idx = []
#     ic = 0
    
#     for I in where_i:
#         jc = 0
#         for J in where_j:
#             Blocks  += [(I[0][0], J[0][1])]
#             loc_idx += [(I[1][0], J[1][1])]
#             sub_idx += [(ic,jc)]
#             jc += 1
#         ic += 1
    
#     unique_blocks = unique_list(Blocks)
#     for blox in unique_blocks:
#         idx = find_list(Blocks, blox)
#         B = block_matrix.Block(blox[0],blox[1])
#         if B is not None:
#             for i in idx:
#                 ii, jj  = loc_idx[i]
#                 ic, jc  = sub_idx[i]
#                 Out[..., ic,jc] = B[..., ii, jj]
#     return Out
 
def find_ele_nested_list(ele, l):
    out = [(i, el.index(ele)) for i, el in 
           enumerate(l) if ele in el]
    return out

def find_unique_blocks_and_indecies(l,i):
    from sys import getsizeof
    #Book-keeping hell
    fried_memory = 0
    n = len(l)
    for j in range(n-1,-1,-1):
        for jj in range(0,j):
            if np.isclose(l[j][..., 0,0],l[jj][..., 0,0]).all():
                if l[j].shape == l[jj].shape:
                    if np.isclose(l[j][0,0],l[jj][0,0]).all():
                        fried_memory+= getsizeof(l[j])
                        del l[j]
                        i[j] = i[jj]
                        break
    s = list(set(sorted(i)))
    i = [s.index(p) for p in i]
    print(str(fried_memory * 10**-9 ) , 'GB freed')
    
    return l,i

def sqrtm_on_KLMij(M):
    Res = np.zeros(M.shape,dtype = complex)
    
    if len(M.shape) == 2:
        return sqrtm(M)
    if len(M.shape) == 3:
        n = len(M[:,0,0])
        for i in range(n):
            Res[i] = sqrtm(M[i])
        return Res
    if len(M.shape) == 4:
        n = len(M[:,0,0,0])
        m = len(M[0,:,0,0])
        for i in range(n):
            for j in range(m):
                Res[i,j] = sqrtm(M[i,j])
        return Res
    if len(M.shape) == 5:
        n = len(M[:,0,0,0,0])
        m = len(M[0,:,0,0,0])
        k = len(M[0,0,:,0,0])
        
        for i in range(n):
            for j in range(m):
                for l in range(k):
                    Res[i,j,l] = sqrtm(M[i,j,l])
        return Res
