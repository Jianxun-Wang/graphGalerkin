from __future__ import print_function
import numpy as np
import pdb
from pyCaMOtk.create_dbc_strct import create_dbc_strct
from pyCaMOtk.create_fem_resjac import create_fem_resjac
import scipy.sparse.linalg as la
#from pymortestbed.optim.nlsys_core import newtraph
#from pymortestbed.linalg import ScipySpLu as linsolv
################################################################################
def solve_fem(fespc, transf_data, elem, 
			  elem_data,ldof2gdof_eqn, 
			  ldof2gdof_var, e2e, spmat, 
			  dbc,Uf0,tol,maxit):
	if dbc is None:
		dbc=create_dbc_strct(np.max(ldof2gdof_var[:]),[],[])

	# Extract information from input
	ndof_var=np.max(ldof2gdof_var[:])+1
	dbc_idx=dbc.dbc_idx
	dbc_val=dbc.dbc_val
	free_idx=dbc.free_idx


	if Uf0 is None:
		Uf0=np.zeros([ndof_var-len(dbc_idx),1])
	if tol is None:
		tol=1e-8
	if maxit is None:
		maxit=10
	fcn=lambda u_:create_fem_resjac(fespc,u_,transf_data,elem,elem_data,ldof2gdof_eqn,ldof2gdof_var,e2e,spmat,dbc)
	# Han Gao create this for test, delete it after all finished
	Uf,info=solve_newtraph_HanGaoTemp(fcn,Uf0,tol,maxit)
	U=np.zeros(ndof_var)
	if len(dbc_idx)==0:
		pass
	else:
		U[dbc_idx]=dbc_val;
	U[free_idx]=Uf[:,0]
	return U, info
	

def solve_newtraph_HanGaoTemp(fcn,x0,tol,maxit):
	maxit=int(maxit)
	r_nrm=np.zeros([1,maxit])
	dx_nrm=np.zeros([1,maxit])
	# Initialize Newton iterations
	x=x0
	R,dR=fcn(x)
	for k in range(maxit):
		print('iteration',str(k))
		nrm=np.max(np.absolute(R[:]))
		r_nrm[0,k]=nrm
		if nrm < tol:
			info={"succ":True,"nit": k,"r_nrm":r_nrm[0,0:k+1],"dx_nrm":dx_nrm[0,0:k]}
			return x, info
		#dx=-np.linalg.solve(dR,R)
		dx=-la.spsolve(dR,R)
		x=x+dx.reshape(x.shape,order='F')
		dx_nrm[0,k]=np.max(np.absolute(dx[:]))
		R,dR=fcn(x)
	info={"succ":False,"nit":maxit,"r_nrm":r_nrm,"dx_nrm":dx_nrm}
	return x, info

	
	