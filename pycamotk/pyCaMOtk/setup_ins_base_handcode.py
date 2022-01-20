import numpy as np
import pdb
class setup_ins_base_handcode(object):
	"""docstring for setup_ins_base_handcode"""
	def __init__(self,ndim,rho,nu,tb,bnd2nbc):
		self.eqn=IncompressibleNavierStokes(ndim)
		self.bnd2nbc=bnd2nbc
		self.vol_pars_fcn=lambda x,el:np.vstack([rho(x, el),
			                                     nu(x, el),
			                                     np.zeros([ndim+1,1])+np.nan])
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack([rho(x,el),
			 										      nu(x,el),
			 										      tb(x,n,bnd,el,fc)])








class IncompressibleNavierStokes(object):
	"""docstring for IncompressibleNavierStokes"""
	def __init__(self,ndim):
		self.ndim=ndim
		self.nvar=ndim+1
		self.srcflux=lambda UQ,pars,x:\
		             eval_ins_base_handcode_srcflux(UQ,pars,x)
		self.bndstvcflux=lambda nbcnbr,UQ,pars,x,n:\
					     eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n)



def eval_ins_base_handcode_srcflux(UQ,pars,x):
	u=UQ[:,0]; q=UQ[:,1:]
	ndim=u.shape[0]-1
	neqn=ndim+1
	ncomp=ndim+1
	rho=pars[0]
	nu=pars[1]
	v=u[0:ndim]
	v=v.reshape([-1,1],order='F')
	p=u[-1]
	dv=q[0:ndim,:]
	S=np.vstack([-rho*dv.dot(v),-np.trace(dv)])
	F=np.vstack([-rho*nu*dv+p*np.eye(ndim),
		         np.zeros([1,ndim])])
	SF= np.hstack([S,F])
	dSFdUQ=np.zeros([neqn,ndim+1,ncomp,ndim+1])
	dSFdUQ[:,0,:,0]=np.vstack([np.hstack([-rho*dv,np.zeros([ndim,1])]), np.zeros([1,ndim+1])])
	for i in range(ndim):
		dSFdUQ[i,0,i,1:]=-rho*v.reshape(dSFdUQ[i,0,i,1:].shape,order='F')
	dSFdUQ[-1,0,0:-1,1:]=np.reshape(-np.eye(ndim),[1,ndim,ndim],order='F')
	dSFdUQ[0:-1,1:,-1,0]=np.eye(ndim)
	for i in range(ndim):
		for j in range(ndim):
			dSFdUQ[i,1+j,i,1+j]=dSFdUQ[i,1+j,i,1+j]-rho*nu
	return SF,dSFdUQ

def eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n):
	nvar=UQ.shape[0]
	ndim=UQ.shape[1]-1
	Ub=UQ[:,0]
	dUb=np.zeros([nvar,nvar,ndim+1])
	dUb[:,:,0]=np.eye(nvar)
	Fn=-pars[-ndim-1:]
	dFn=np.zeros([nvar,nvar,ndim+1])
	return Ub,dUb,Fn,dFn







