from __future__ import print_function
import numpy as np
import pdb
################################################################################
def intg_elem_claw_extface(Ue,transf_data,elem,elem_data,e):
	[neqn_per_elem,neqn,ndimP1,nqf,nf]=elem.Tvf_eqn_ref.shape
	[nvar_per_elem,nvar,_,_,_]=elem.Tvf_var_ref.shape
	ndim=ndimP1-1
	wqf=elem.wqf
	sigf=transf_data.sigf[:,:,e]
	nbcnbr=elem_data.nbcnbr[:,e]
	Re=np.zeros([neqn_per_elem,1])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem])
	wf=wqf[:].reshape(len(wqf),1)*sigf
	for f in range(nf):
		if np.isnan(nbcnbr[f]):
			continue
		Tvar=np.reshape(elem_data.Tvf_var_phys[:,:,:,:,f,e],
						[nvar_per_elem,nvar*(ndim+1)*nqf],order='F')
		UQqf=np.reshape(Tvar.T.dot(Ue),[nvar,ndim+1,nqf],order='F')
		for k in range(nqf):
			x=transf_data.xqf[:,k,f,e]
			n=transf_data.n[:,k,f,e]
			Teqn=elem_data.Tvf_eqn_phys[:,:,0,k,f,e]
			Tvar=np.reshape(elem_data.Tvf_var_phys[:,:,:,k,f,e],
				            [nvar_per_elem,nvar*(ndim+1)],order='F')
			pars=elem_data.bnd_pars[:,k,f,e]
			_,_,Fb,dFbdU=elem.eqn.bndstvcflux(nbcnbr[f],UQqf[:,:,k],pars,x,n)
			dFbdU=np.reshape(dFbdU,[neqn,nvar*(ndim+1)],order='F')
			Re=Re+wf[k,f]*Teqn.dot(Fb).reshape(Re.shape,order='F')
			dRe=dRe+wf[k,f]*Teqn.dot(dFbdU.dot(Tvar.T))
	return Re,dRe

