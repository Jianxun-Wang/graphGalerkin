from pyCaMOtk.create_ldof2gdof_dg import create_ldof2gdof_dg
from pyCaMOtk.create_ldof2gdof_cg import create_ldof2gdof_cg
from pyCaMOtk.create_elem_strct import create_elem_strct
from pyCaMOtk.create_elem_data import create_elem_data
from pyCaMOtk.lfcnsp import LocalFunctionSpace
from pyCaMOtk.create_sparsity_strct_dg import create_sparsity_strct_dg
from pyCaMOtk.create_dbc_strct import create_dbc_strct
import numpy as np
import pdb
################################################################################
class create_femsp_dg(object):
	"""docstring for create_femsp_dg"""
	def __init__(self,prob,msh,ptest,ptrial):
		self.prob=prob; self.msh=msh
		self.ptest=ptest; self.ptrial=ptrial
		# Extract information from input
		self.etype=self.msh.etype; self.ndim=msh.xcg.shape[0]
		self.nelem=self.msh.e2vcg.shape[1]; self.e2e=self.msh.e2e
		self.transf_data=self.msh.transfdatacontiguous
		self.nquad_per_dim=self.msh.lfcnsp.nq0
		self.neqn=self.prob.eqn.neqn
		self.nvar=self.neqn
		if self.etype=='hcube':
			self.polysp='Q'
		elif self.etype=='simp' or self.etype=='simplex':
			self.polysp='P'
		else:
			raise ValueError('Only support simplex or hcube!')

		self.lfcnsp_eqn=LocalFunctionSpace(self.polysp,
			                               self.ndim,
			                               self.ptest,
			                               'gl',
			                               self.nquad_per_dim)
		self.lfcnsp_var=LocalFunctionSpace(self.polysp,
										   self.ndim,
										   self.ptrial,
										   'gl',
										   self.nquad_per_dim)
		self.lfcnsp_msh=self.msh.lfcnsp
		self.nv_eqn=self.lfcnsp_eqn.Qvv.shape[0]
		self.nv_var=self.lfcnsp_var.Qvv.shape[0]

		self.ldof2gdof_eqn=create_ldof2gdof_dg(self.neqn,self.nv_eqn,self.nelem)
		self.ldof2gdof_var=create_ldof2gdof_dg(self.nvar,self.nv_var,self.nelem)
		self.ldof2gdof_msh=create_ldof2gdof_cg(self.ndim,self.msh.e2vcg)

		self.elem=create_elem_strct(self.prob.eqn,
			                        self.lfcnsp_eqn,
			                        self.lfcnsp_var)
		self.elem_data=create_elem_data(self.elem.Tv_eqn_ref, 
										self.elem.Tvf_eqn_ref, 
                             			self.elem.Tv_var_ref, 
                             			self.elem.Tvf_var_ref, 
                             			self.e2e,self.transf_data,None, 
                             			self.prob.vol_pars_fcn,
                             			self.prob.bnd_pars_fcn,None, 
                             			self.prob.bnd2nbc)
		self.spmat=create_sparsity_strct_dg(self.ldof2gdof_eqn.ldof2gdof,
			                                self.ldof2gdof_var.ldof2gdof,
			                                self.e2e)
		self.ndof=len(self.ldof2gdof_var.ldof2gdof.flatten())
		self.dbc=create_dbc_strct(self.ndof,[],[])




		