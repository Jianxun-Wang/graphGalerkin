import numpy as np
import pdb
class elem_basis(object):
	"""docstring for elem_basis"""
	def __init__(self, nc, Qv, Qvf):
		self.nc=nc
		self.Qv=Qv
		self.Qvf=Qvf

		# extract basis info
		self.ndim=self.Qv.shape[1]-1
		self.nv=self.Qv.shape[0]
		self.nq=self.Qv.shape[2]
		self.nqf=self.Qvf.shape[2]
		self.nf=self.Qvf.shape[3]

		# solution basis
		self.Tv=np.zeros((self.nv*self.nc,self.nc,self.ndim+1,self.nq))
		self.Tvf=np.zeros((self.nv*self.nc,self.nc,self.ndim+1,self.nqf,self.nf))
		for k in range(self.nc):
			id_=[k+i*self.nc for i in range(self.nv)]
			self.Tv[id_,k,:,:]=np.copy(self.Qv)
			self.Tvf[id_,k,:,:,:]=np.copy(self.Qvf)




class elem_basis_mixed2(object):
	"""docstring for elem_basis"""
	def __init__(self, nc1, Qv1, Qvf1,
		               nc2, Qv2, Qvf2):
		self.nc1=nc1
		self.nc2=nc2
		self.Qv1=Qv1
		self.Qv2=Qv2
		self.Qvf1=Qvf1
		self.Qvf2=Qvf2
		self.nc=self.nc1+self.nc2
		

		# extract basis info
		self.ndim=self.Qv1.shape[1]-1
		self.nv1=self.Qv1.shape[0]
		self.nq=self.Qv1.shape[2]
		self.nqf=self.Qvf1.shape[2]
		self.nf=self.Qvf1.shape[3]
		self.nv2=self.Qv2.shape[0]
		self.ndof_per_elem=self.nv1*self.nc1+self.nv2*self.nc2
		
		self.Tv=np.zeros([self.ndof_per_elem,self.nc,self.ndim+1,self.nq])
		self.Tvf=np.zeros([self.ndof_per_elem,self.nc,self.ndim+1,self.nqf,self.nf])
		
		for k in range(self.nc1):
			id_=[k+i*self.nc1 for i in range(self.nv1)]
			self.Tv[id_, k, :, :]=self.Qv1
			self.Tvf[id_, k, :, :, :]=self.Qvf1

		for k in range(self.nc2):
			id_=[self.nv1*self.nc1+k+i*self.nc2 for i in range(self.nv2)]
			self.Tv[id_,self.nc1+k,:,:]=self.Qv2
			self.Tvf[id_,self.nc1+k,:,:,:]=self.Qvf2



