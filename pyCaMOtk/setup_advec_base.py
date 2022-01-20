import numpy as np
from pyCaMOtk.AdvectionBaseHandcode import AdvectionBaseHandcode
class setup_advec_base(object):
	"""docstring for setup_advec_base"""
	def __init__(self,ndim,beta,f,b,Ub,smooth_par,bnd2nbc,which_numflux):
		self.ndim=ndim
		self.beta=beta
		self.f=f
		self.b=b
		self.Ub=Ub
		self.smooth_par=smooth_par
		self.bnd2nbc=bnd2nbc
		self.which_numflux=which_numflux

		if self.beta is None or self.beta==[]:
			self.beta=lambda x,el:np.ones([ndim,1])
		if self.f is None or self.f==[]:
			self.f=lambda x,el:0
		if self.b is None or self.b==[]:
			self.b=lambda x,el:0
		if self.Ub is None or self.Ub==[]:
			self.Ub=lambda x,n,bnd,el,fc: 0
		if self.smooth_par is None or self.smooth_par==[]:
			self.smooth_par=0

		if self.ndim==2:
			self.eqn=AdvectionBaseHandcode()
		else:
			raise ValueError('Only support 2d AdvectionBaseHandcode!')

		self.vol_pars_fcn=\
		lambda x,el: np.vstack([self.beta(x, el),self.f(x, el),
			                    self.b(x,el),np.nan,self.smooth_par])
		self.bnd_pars_fcn=\
		lambda x,n,bnd,el,fc: np.vstack([self.beta(x,el),
			                             self.f(x,el),
			                             self.b(x,el),
			                             self.Ub(x,n,bnd,el,fc),
			                             self.smooth_par])







		pass
		