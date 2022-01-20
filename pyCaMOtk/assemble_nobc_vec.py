import numpy as np
import pdb

def assemble_nobc_vec(Fe,ldof2gdof_eqn):
	ndof=np.max(ldof2gdof_eqn[:])+1
	nelem=Fe.shape[1]
	F=np.zeros(ndof)
	for e in range(nelem):
		idx=ldof2gdof_eqn[:,e]
		F[idx]=F[idx]+Fe[:,e]
	return F
