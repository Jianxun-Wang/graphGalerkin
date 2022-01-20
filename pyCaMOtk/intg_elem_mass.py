import numpy as np
import pdb

class intg_elem_mass(object):
	"""docstring for intg_elem_mass"""
	def __init__(self,Teqn,Tvar,wq,detG,coeff=None):
		self.Teqn=Teqn
		self.Tvar=Tvar
		self.wq=wq
		self.detG=detG
		
		if coeff is None:
			self.coeff=np.ones(self.wq.shape)
		else:
			self.coeff=coeff

		if len(self.Teqn.shape)==2:
			self.Teqn=np.reshape(self.Teqn,[self.Teqn.shape[0],1,self.Teqn.shape[1]],order='F')

		if len(self.Tvar.shape)==2:
			self.Tvar=np.reshape(self.Tvar,[self.Tvar.shape[0],1,self.Tvar.shape[1]],order='F')

		self.nvar_per_elem,self.nvar,self.nq=self.Tvar.shape
		self.neqn_per_elem,self.neqn,garbage=self.Teqn.shape

		assert self.nvar==self.neqn,'Mass matrix not well-defined when nvar ~= neqn!'

		# Begin mass matrix assembling
		self.w_sqrt=np.sqrt(self.wq*self.detG*self.coeff)

		# Modify bases to include weights
		self.Tvar=self.Tvar*self.w_sqrt.reshape(1,1,self.nq, order='F')
		self.Tvar=self.Tvar.reshape(self.nvar_per_elem,self.nvar*self.nq,order='F')
		self.Teqn=self.Teqn*self.w_sqrt.reshape(1,1,self.nq, order='F')
		self.Teqn=self.Teqn.reshape(self.neqn_per_elem,self.neqn*self.nq,order='F')
		self.Me=np.matmul(self.Teqn,self.Tvar.T)

		



