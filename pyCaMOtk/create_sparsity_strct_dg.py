from __future__ import print_function
import numpy as np
import pdb
class create_sparsity_strct_dg(object):
	"""docstring for create_sparsity_strct_dg"""
	def __init__(self,ldof2gdof1,ldof2gdof2,e2e):
		[self.nf,_,self.nelem]=e2e.shape
		self.ldof2gdof1=ldof2gdof1; self.ldof2gdof2=ldof2gdof2
		self.e2e=e2e; self.e2e0=np.squeeze(self.e2e[:,0,:])
		self.nldof1=self.ldof2gdof1.shape[0]
		self.nldof2=self.ldof2gdof2.shape[0]
		self.nnz_blk=self.nldof1*self.nldof2
		self.nnz=self.nnz_blk*(self.nelem+len([i for i in \
							   self.e2e0.flatten() if not np.isnan(i)]))
		self.cooidx=np.zeros([self.nnz,2])+np.nan
		self.lmat2gmat=np.nan+np.zeros([self.nldof1,self.nldof2,
								 self.nf+1,self.nelem])
		s0=0
		for e in range(self.nelem):
			for k in range(-1,self.nf):
				if k>-1 and any(np.isnan(e2e[k,:,e])):
					continue
				if k==-1:
					ep=e
				else:
					ep=int(self.e2e[k,0,e])
				idx1=self.ldof2gdof1[:,e]
				idx2=self.ldof2gdof2[:,ep]
				[jcol0,irow0]=np.meshgrid(idx2,idx1)
				sF=s0+self.nldof1*self.nldof2
				self.cooidx[s0:sF,0]=irow0.flatten('F')
				self.cooidx[s0:sF,1]=jcol0.flatten('F')
				self.lmat2gmat[:,:,k+1,e]=np.reshape(range(s0,sF),
					                                [self.nldof1,self.nldof2],
					                                 order='F')
				s0=sF
		self.lmat2gmat=self.lmat2gmat
		self.cooidx=self.cooidx
		

		