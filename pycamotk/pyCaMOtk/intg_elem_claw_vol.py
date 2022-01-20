from __future__ import print_function
import numpy as np
import pdb
################################################################################
def intg_elem_claw_vol(Ue,transf_data,elem,elem_data,e):
	# Han Gao modify this for one more arg: e (element index)
	# Extract information from input : sizes
	[neqn_per_elem,neqn,ndimP1,nq]=elem.Tv_eqn_ref.shape
	[nvar_per_elem,nvar,_,_]=elem.Tv_var_ref.shape
	ndim=ndimP1-1
	wq=elem.wq
	detG=transf_data.detG[:,e]
	Tvar=elem_data.Tv_var_phys[:,:,:,:,e].reshape([nvar_per_elem,nvar*(ndim+1)*nq],
												  order='F')
	Re=np.zeros([neqn_per_elem,1])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem])
	UQq=np.reshape(Tvar.T.dot(Ue),[nvar,ndim+1,nq],order='F')
	w=wq*detG
	for k in range(nq):
		Teqn=elem_data.Tv_eqn_phys[:,:,:,k,e].reshape([neqn_per_elem,neqn*(ndim+1)],
			                                      order='F')
		Tvar=elem_data.Tv_var_phys[:,:,:,k,e].reshape([nvar_per_elem,nvar*(ndim+1)],
			                                      order='F')
		x=transf_data.xq[:,k,e]
		pars=elem_data.vol_pars[:,k,e]
		SF, dSFdU=elem.eqn.srcflux(UQq[:,:,k],pars,x)
		SF=SF.flatten(order='F')
		dSFdU=np.reshape(dSFdU,[neqn*(ndim+1),nvar*(ndim+1)],order='F')
		Re=Re-w[k]*Teqn.dot(SF).reshape(Re.shape,order='F')
		dRe=dRe-w[k]*(Teqn.dot(dSFdU.dot(Tvar.T)))
	return Re, dRe
		


