from __future__ import print_function
import numpy as np
import pdb
from pyCaMOtk.intg_elem_claw_vol import intg_elem_claw_vol
from pyCaMOtk.intg_elem_claw_extface import intg_elem_claw_extface
from pyCaMOtk.intg_elem_claw_intface import intg_elem_claw_intface

def eval_unassembled_resjac_claw_dg(U,transf_data,elem,elem_data,ldof2gdof_var,e2e):
	[nf,_,nelem]=e2e.shape
	neqn_per_elem=elem.Tv_eqn_ref.shape[0]
	nvar_per_elem=elem.Tv_var_ref.shape[0]
	if U is None or len(U)==0 or U==[]:
		ndof_var=np.max(ldof2gdof_var[:])+1
		U=np.zeros([ndof_var,1])
	Uef=np.zeros([nvar_per_elem,nf+1])
	Re=np.zeros([neqn_per_elem,nelem])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem,nf+1,nelem])
	for e in range(nelem):
		Ue=U[ldof2gdof_var[:,e]]
		Uef[:,0]=Ue.reshape(Uef[:,0].shape,order='F')
		for f in range(nf):
			if np.isnan(e2e[f,0,e]):
				Uef[:,1+f]=np.nan
				continue
			ep=int(e2e[f,0,e])
			Uef[:,1+f]=U[ldof2gdof_var[:,ep]].reshape(Uef[:,1+f].shape,order='F')
		elst=[e]
		for i in range(len(e2e[:,0,e].flatten())):
			if np.isnan(e2e[i,0,e]):
				elst.append(e)
			else:
				elst.append(int(e2e[i,0,e]))
		Re0_,dRe0_=intg_elem_claw_vol(Ue,transf_data,elem,elem_data,e)
		Re1_,dRe1_=intg_elem_claw_extface(Ue,transf_data,elem,elem_data,e)
		Re2_,dRe2_=intg_elem_claw_intface(Uef,transf_data,elem,elem_data,e2e,e,elst)
		Re[:,e]=(Re0_+Re1_+Re2_).reshape(Re[:,e].shape,order='F')
		dRe[:,:,0,e]=(dRe0_+dRe1_+dRe2_[:,:,0]).reshape(dRe[:,:,0,e].shape,order='F')
		for f in range(nf):
			if np.isnan(e2e[f,0,e]):
				continue
			dRe[:,:,1+f,e]=dRe[:,:,1+f,e]+dRe2_[:,:,1+f]
	return Re,dRe