import numpy as np
import pdb
################################################################################
class AdvectionBaseHandcode(object):
	"""docstring for Advection2dBase"""
	def __init__(self):
		self.neqn=1; self.nvar=1
		self.srcflux=lambda UQ,pars,X:eval_advec_base_handcode_srcflux_local\
		                              (UQ,pars,X)
		self.numflux=lambda UQi,parsI,UQo,parsO,x,n:\
		             eval_advec_base_handcode_numflux_upwind\
		             (UQi,parsI,UQo,parsO,x,n)
		self.bndstvcflux=lambda nbcnbr,UQ,pars,x,n:\
					     eval_advec_base_handcode_bndstvcflux\
					     (nbcnbr,UQ,pars,x,n)

		self.eval_advec_base_rankhug=lambda UQi,parsI,UQo,parsO,x,n:\
		                             eval_advec_base_handcode_rankhug\
		                             (UQi,parsI,UQo,parsO,x,n)



def eval_advec_base_handcode_srcflux_local(UQ,pars,X):
	SF,dSF=eval_advec_base_handcode_srcflux(UQ,pars,X)
	return SF,dSF

def eval_advec_base_handcode_bndstvcflux(nbcnbr,UQ,pars,x,n):
	Ub,dUbdUQ,Fn,dFndUQ=eval_advec_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n)
	return Ub,dUbdUQ,Fn,dFndUQ

def eval_advec_base_handcode_srcflux(UQ,pars,x):
	# Define information regarding size of the system
	neqn=1; nvar=1
	# Extract information from input
	u=UQ[0,0];q=UQ[0,1:] #primary and derivative
	ndim=q.shape[0]
	beta=pars[0:ndim]
	f=pars[ndim]
	b=pars[ndim+1]
	# Define flux and source
	SF=np.asarray(np.hstack([f-u*b,u*beta[:].T]))
	dSFdUQ=np.zeros([neqn,ndim+1,nvar,ndim+1])
	dSFdUQ[0,0,0,0]=-b # dSdu
	dSFdUQ[0,1:,0,0]=beta # dFdu
	return SF,dSFdUQ

def eval_advec_base_handcode_numflux_upwind(UQi,parsI,UQo,parsO,x,n):
	ndim=np.max(x.shape)
	neqn=1; nvar=1
	ui=UQi[:,0]; uo = UQo[:,0]
	beta=parsI[0:ndim]
	bn=beta.T.dot(n)
	if bn>=0:
		Fn=ui.dot(bn)
	else:
		Fn=uo.dot(bn)

	dFndUQi=np.zeros([neqn,nvar,ndim+1])
	dFNdUQo=np.zeros([neqn,nvar,ndim+1])

	if bn >= 0:
		dFndUQi[0,0,0]=bn
	else:
		dFNdUQo[0,0,0]=bn
	return Fn,dFndUQi,dFNdUQo



def eval_advec_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n):
	neqn=1; nvar=1
	ndim=UQ.shape[1]-1
	beta=pars[0:ndim]
	Ub=pars[ndim+2]
	dUb=np.zeros([nvar,nvar,ndim+1])
	bn=beta.T.dot(n)
	ui=UQ[:,0];uo=Ub
	if bn>=0:
		try:
			Fn=ui.dot(bn)
		except:
			Fn=ui*bn
	else:
		try:
			Fn=uo.dot(bn)
		except:
			Fn=uo*bn
	dFn=np.zeros([neqn,nvar,ndim+1])
	if bn>=0:
		dFn[0,0,0]=bn
	return Ub,dUb,Fn,dFn

def eval_advec_base_handcode_rankhug(UQi,parsI,UQo,parsO,x,n):
	neqn=1; nvar=1; ndim=np.max(x.shape)
	ui=UQi[0];uo=UQo[0]
	betaIn=parsI[0:ndim].T.dot(n)
	betaOn=parsO[0:ndim].T.dot(n)
	dFjdUQi=np.zeros([neqn,nvar,ndim+1])
	dFjdUQo=np.zeros([neqn,nvar,ndim+1])
	dFjdUQi[0,0,0]=betaIn
	dFjdUQo[0,0,0]=-betaOn
	return Fj,dFjdUQi,dFjdUQo







			
