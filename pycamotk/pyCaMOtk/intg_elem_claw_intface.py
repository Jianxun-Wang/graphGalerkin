from __future__ import print_function
import numpy as np
import pdb
def intg_elem_claw_intface(Uef,transf_data,elem,elem_data,e2e,e,elst):
	e2e=e2e[:,:,e]
	[neqn_per_elem,neqn,ndimP1,nqf,nf]=elem.Tvf_eqn_ref.shape
	[nvar_per_elem,nvar,_,_,_]=elem.Tvf_var_ref.shape
	ndim=ndimP1-1; wqf=elem.wqf; perm=transf_data.perm[:,:,e]
	sigf=transf_data.sigf[:,:,e]
	Re=np.zeros([neqn_per_elem,1])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem,nf+1])
	wf=wqf.reshape([-1,1],order='F')*sigf
	for f in range(nf):
		if np.isnan(e2e[f,0]):
			continue
		fp=int(e2e[f,1])
		TvfI=np.reshape(elem_data.Tvf_var_phys[:,:,:,:,f,elst[0]],
						[nvar_per_elem,nvar*(ndim+1)*nqf],order='F')
		UQqfI=np.reshape(TvfI.T.dot(Uef[:,0]),[nvar,ndim+1,nqf],order='F')
		TvfO=np.reshape(elem_data.Tvf_var_phys[:,:,:,:,fp,elst[(1+f)]],
						[nvar_per_elem,nvar*(ndim+1)*nqf],order='F')
		UQqfO=np.reshape(TvfO.T.dot(Uef[:,1+f]),[nvar,ndim+1,nqf],order='F')
		UQqfO=UQqfO[:,:,tuple(perm[:,f].astype('int'))]
		for k in range(nqf):
			x=transf_data.xqf[:,k,f,e]
			n=transf_data.n[:,k,f,e]
			parsI=elem_data.face_pars[:,k,f,elst[0]]
			parsO=elem_data.face_pars[:,int(perm[k,f]),fp,elst[1+f]]
			Fs,dFsdUQI,dFsdUQO=elem.eqn.numflux(UQqfI[:,:,k],parsI,UQqfO[:,:,k],parsO,x,n)
			dFsdUQI=np.reshape(dFsdUQI,[neqn,nvar*(ndim+1)],order='F')
			dFsdUQO=np.reshape(dFsdUQO,[neqn,nvar*(ndim+1)],order='F')
			TvfI0=elem_data.Tvf_eqn_phys[:,:,0,k,f,elst[0]]
			TvfI=np.reshape(elem_data.Tvf_var_phys[:,:,:,k,f,elst[0]],
							[nvar_per_elem,nvar*(ndim+1)],order='F')
			TvfO=np.reshape(elem_data.Tvf_var_phys[:,:,:,int(perm[k,f]),fp,elst[1+f]],
							[nvar_per_elem,nvar*(ndim+1)],order='F')
			Re=Re+np.reshape(wf[k,f]*TvfI0.dot(Fs),Re.shape,order='F')
			dRe[:,:,0]=dRe[:,:,0]+wf[k,f]*TvfI0.dot(dFsdUQI.dot(TvfI.T))
			dRe[:,:,1+f]=dRe[:,:,1+f]+wf[k,f]*TvfI0.dot(dFsdUQO.dot(TvfO.T))
	return Re, dRe