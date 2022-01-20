import numpy as np
import pdb
from pyCaMOtk.eval_unassembled_resjac_claw_dg import eval_unassembled_resjac_claw_dg
from pyCaMOtk.eval_unassembled_resjac_claw_cg import eval_unassembled_resjac_claw_cg
from pyCaMOtk.assemble_nobc_mat import assemble_nobc_mat, assemble_nobc_mat_dg
from pyCaMOtk.assemble_nobc_vec import assemble_nobc_vec
################################################################################
def create_fem_resjac(fespc,Uf,transf_data,elem,elem_data,
                      ldof2gdof_eqn,ldof2gdof_var,e2e,spmat,dbc):
	# Extract information from input
	ndof_var=np.max(ldof2gdof_var[:])+1
	dbc_idx=dbc.dbc_idx
	dbc_val=dbc.dbc_val
	free_idx=dbc.free_idx

	U=np.zeros([ndof_var,1])
	if dbc_val==[] or dbc_val is None:
		pass
	else:
		U[dbc_idx,0]=dbc_val
	U[free_idx,0]=Uf[:,0]
	if fespc=='cg':
		Re,dRe=eval_unassembled_resjac_claw_cg(U,transf_data,elem,elem_data,
			                                   ldof2gdof_var)
		dR=assemble_nobc_mat(dRe,spmat.cooidx,spmat.lmat2gmat)
	elif fespc=='dg':
		Re,dRe=eval_unassembled_resjac_claw_dg(U,transf_data,elem,elem_data,
			                                   ldof2gdof_var,e2e)
		dR=assemble_nobc_mat_dg(dRe,spmat.cooidx,spmat.lmat2gmat)
	else:
		raise ValueError('Only support dg or cg')
		
	R=assemble_nobc_vec(Re,ldof2gdof_eqn)
	Rf=R[free_idx]
	dRf=dR.tocsr()[free_idx,:]
	dRf=dRf.tocsr()[:,free_idx]
	return Rf,dRf
	