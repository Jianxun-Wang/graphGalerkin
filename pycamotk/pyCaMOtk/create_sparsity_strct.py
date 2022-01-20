import numpy as np
import pdb
class create_sparsity_strct(object):
	"""docstring for create_sparsity_strct"""
	def __init__(self,*args):
		self.args=args
		self.nldof2gdof=len(self.args)
		if self.nldof2gdof==1:
			self.ldof2gdof1=np.copy(self.args[0])
			self.ldof2gdof2=np.copy(self.args[0])
		elif self.nldof2gdof==2:
			self.ldof2gdof1=np.copy(self.args[0])
			self.ldof2gdof2=np.copy(self.args[1])
		else:
			raise ValueError('Only support the number of ldof2gdof be 1 or 2!')
		self.nldof1,self.nelem1=self.ldof2gdof1.shape
		self.nldof2,self.nelem2=self.ldof2gdof2.shape

		assert self.nelem1==self.nelem2,'Both have same nubmer of elements!'

		self.nelem=self.nelem1

		self.cooidx_rep=np.zeros([self.nldof1*self.nldof2*self.nelem,2])

		for e in range(self.nelem):
			idx1=self.ldof2gdof1[:,e].reshape(-1,)
			idx2=self.ldof2gdof2[:,e].reshape(-1,)
			[jcol0,irow0]=np.meshgrid(idx2,idx1)
			self.cooidx_rep[self.nldof1*self.nldof2*e:self.nldof1*self.nldof2*(e+1),0]=irow0.reshape(-1,order='F')
			self.cooidx_rep[self.nldof1*self.nldof2*e:self.nldof1*self.nldof2*(e+1),1]=jcol0.reshape(-1,order='F')

		cooidx,idx_,lmat2gmat=np.unique(self.cooidx_rep,return_index=True,return_inverse=True, axis=0)
		idx_map=np.argsort(idx_)
		self.cooidx=self.cooidx_rep[np.sort(idx_),:]
		self.lmat2gmat=[np.argwhere(idx_map==lmat2gmat[i])[0,0] for i in range(len(lmat2gmat))]
		self.lmat2gmat=np.asarray(self.lmat2gmat)
		self.lmat2gmat=self.lmat2gmat.reshape(self.nldof1,self.nldof2,self.nelem,order='F')
		
		
		

		