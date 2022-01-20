import numpy as np
import pdb
class LinearEllipticScalarBaseHandcode(object):
	"""docstring for LinearEllipticScalarBaseHandcode"""
	def __init__(self):
		self.neqn=1
		self.nvar=1
		self.ncomp=1

	def srcflux(self,UQ,pars,x):
		"""
		eval_linelptc_base_handcode_srcflux
		"""
		# Extract information from input
		q=UQ[0,1:]
		self.ndim=len(q)
		k=np.reshape(pars[0:self.ndim**2],
			         (self.ndim,self.ndim),order='F')
		f=pars[self.ndim**2]
		#pdb.set_trace()

		# Define flux and source
		SF=np.zeros(self.ndim+1)
		SF[0]=f
		SF[1:]=-1*np.matmul(k,q)

		# Define partial derivative
		dSFdU=np.zeros([self.neqn, self.ndim+1, self.ncomp,self.ndim+1]);
		dSFdU[:,1:,:,1:]=np.reshape(-1*k, 
			                        [self.neqn, self.ndim,
			                         self.ncomp,self.ndim],order='F')
		return SF, dSFdU

	def bndstvcflux(self,nbcnbr,UQ,pars,x,n):
		nvar=UQ.shape[0]
		ndim=UQ.shape[1]-1

		Ub=UQ[:,0]
		dUb=np.zeros([nvar,nvar,self.ndim+1])
		dUb[:,:,0]=np.eye(nvar)
		Fn=pars[ndim**2+1]
		dFn=np.zeros([nvar,nvar,self.ndim+1])
		return Ub,dUb,Fn,dFn

