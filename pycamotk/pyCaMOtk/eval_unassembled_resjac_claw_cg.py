from __future__ import print_function
import numpy as np
import pdb
from pyCaMOtk.intg_elem_claw_vol import intg_elem_claw_vol
from pyCaMOtk.intg_elem_claw_extface import intg_elem_claw_extface
################################################################################
def eval_unassembled_resjac_claw_cg(U,transf_data,elem,elem_data,ldof2gdof_var):
	nelem=elem_data.nelem
	neqn_per_elem=elem.Tv_eqn_ref.shape[0]
	nvar_per_elem=elem.Tv_var_ref.shape[0]
	if U is None or len(U)==0 or U==[]:
		ndof_var=np.max(ldof2gdof_var[:])+1
		U=np.zeros([ndof_var,1])
	Re=np.zeros([neqn_per_elem,nelem])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem,nelem])
	for e in range(nelem):
		Ue=U[ldof2gdof_var[:,e]]
		Re0_,dRe0_=intg_elem_claw_vol(Ue,transf_data,elem,elem_data,e)
		Re1_,dRe1_=intg_elem_claw_extface(Ue,transf_data,elem,elem_data,e)
		Re[:,e]=(Re0_+Re1_).reshape(Re[:,e].shape,order='F')
		dRe[:,:,e]=(dRe0_+dRe1_).reshape(dRe[:,:,e].shape,order='F')
	return Re,dRe
	