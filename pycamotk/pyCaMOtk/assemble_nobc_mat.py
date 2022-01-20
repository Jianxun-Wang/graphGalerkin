import numpy as np
import pdb
from scipy import sparse
def assemble_nobc_mat(Me,cooidx,lmat2gmat):
	nnz=cooidx.shape[0]
	cooidx=cooidx.astype('int')
	Mval=np.zeros(nnz)
	nelem=Me.shape[2]
	for e in range(nelem):
		idx=lmat2gmat[:,:,e]
		Mval[idx]=Mval[idx]+Me[:,:,e]
	M=sparse.coo_matrix((Mval,(cooidx[:,0],cooidx[:,1])))
	return M

def assemble_nobc_mat_dg(Me,cooidx,lmat2gmat):
	if len(Me.shape)==3:
		Me=np.reshape(Me,[Me.shape[0],
			              Me.shape[1],
			              1,Me.shape[2]])
	nnz=cooidx.shape[0]
	Mval=np.zeros(nnz)
	[_,_,nfp1,nelem]=Me.shape
	for e in range(nelem):
		for f in range(nfp1):
			idx=lmat2gmat[:,:,f,e].flatten(order='F')
			if any([np.isnan(i) for i in idx.flatten()]):
				continue
			Me_=Me[:,:,f,e]
			Mval[idx.astype('int')]=Mval[idx.astype('int')]+Me_.flatten(order='F')
	M=sparse.coo_matrix((Mval,(cooidx[:,0].astype('int'),
		                       cooidx[:,1].astype('int'))))
	return M


