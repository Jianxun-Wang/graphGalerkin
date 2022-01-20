import numpy as np
import pdb
class setup_linelast_base_handcode(object):
	"""docstring for setup_linelast_base_handcode"""
	def __init__(self,ndim,lam,mu,f,tb,bnd2nbc):
		self.bnd2nbc=bnd2nbc
		self.eqn=LinearElasticityBaseHandcode(ndim)
		self.vol_pars_fcn=lambda x, el: np.vstack((lam(x,el),
			                                      mu(x,el),
			                                      f(x,el),
			                                      np.zeros([ndim,1])+np.nan))
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack((lam(x, el),
			                                              mu(x, el),
			                                              f(x, el),
			                                              tb(x, n, bnd, el, fc)))

		

class LinearElasticityBaseHandcode(object):
	"""docstring for LinearElasticityBaseHandcode"""
	def __init__(self,ndim):
		self.neqn=ndim
		self.nvar=ndim
		self.bndstvcflux=\
		lambda nbcnbr, UQ, pars, x, n:\
		eval_linelast_base_handcode_bndstvc_intr_bndflux_pars(UQ, pars, x, n)
		self.srcflux=lambda UQ,pars,x:\
		eval_linelast_base_handcode_srcflux(UQ, pars, x)

def eval_linelast_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n):
	nvar=UQ.shape[0]
	ndim=UQ.shape[1]-1

	Ub=UQ[:,0]
	dUb=np.zeros([nvar,nvar,ndim+1])
	dUb[:,:,0]=np.eye(nvar)
	Fn=-pars[-ndim:]
	dFn=np.zeros([nvar,nvar,ndim+1])
	return Ub,dUb,Fn,dFn

def eval_linelast_base_handcode_srcflux(UQ, pars, x):
	q=UQ[:,1:]
	ndim=q.shape[0]
	# Define information regarding size of the system
	neqn=ndim
	ncomp=ndim

	# Extract parameters
	lam=pars[0]
	mu=pars[1]
	f=pars[2:2+ndim]
	F=-lam*np.trace(q)*(np.eye(ndim))-mu*(q+q.T)
	S=f.reshape([ndim,1],order='F')
	SF=np.hstack((S,F))
	dSFdU=np.zeros([neqn,ndim+1,ncomp,ndim+1]);
	for i in range(ndim):
		for j in range(ndim):
			dSFdU[i,1+i,j,1+j]=dSFdU[i,1+i,j,1+j]-lam
			dSFdU[i,1+j,i,1+j]=dSFdU[i,1+j,i,1+j]-mu
			dSFdU[i,1+j,j,1+i]=dSFdU[i,1+j,j,1+i]-mu
	return SF, dSFdU


		