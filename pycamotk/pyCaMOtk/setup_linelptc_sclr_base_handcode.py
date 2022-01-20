import numpy as np
from pyCaMOtk.LinearEllipticScalarBaseHandcode import LinearEllipticScalarBaseHandcode
import pdb

class setup_linelptc_sclr_base_handcode(object):
	"""docstring for setup_linelptc_sclr_base_handcode"""
	def __init__(self,ndim,K,f,Qb,bnd2nbc):
		self.ndim=ndim
		self.K=K
		self.f=f
		self.Qb=Qb
		self.bnd2nbc=bnd2nbc

		self.I=np.eye(self.ndim)
		if self.K==None:
			self.K=lambda x,el: self.I.reshape(self.ndim**2,1,order='F')
		if self.f==None:
			self.f=lambda x,el: 0
		if self.Qb==None:
			self.Qb=lambda x,n,bnd,el,fc: 0

		self.eqn=LinearEllipticScalarBaseHandcode()
		self.vol_pars_fcn=lambda x,el:np.vstack((self.K(x, el),self.f(x, el),np.nan))
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack((self.K(x,el),
			                                              self.f(x,el),
			                                              self.Qb(x,n,bnd,el,fc)))

		
