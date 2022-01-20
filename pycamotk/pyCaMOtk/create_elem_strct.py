from pyCaMOtk.create_elem_basis import elem_basis,elem_basis_mixed2
import pdb 
class create_elem_strct(object):
	"""docstring for create_elem_strct"""
	def __init__(self,eqn,lfcnsp_eqn,lfcnsp_var):
		self.eqn=eqn
		self.lfcnsp_eqn=lfcnsp_eqn
		self.lfcnsp_var=lfcnsp_var
		self.eqn_elem=elem_basis(self.eqn.neqn, 
			                     self.lfcnsp_eqn.Qvv,
			                     self.lfcnsp_eqn.Qvf)

		self.var_elem=elem_basis(self.eqn.nvar,
			                     self.lfcnsp_var.Qvv,
			                     self.lfcnsp_var.Qvf)

		self.Tv_eqn_ref=self.eqn_elem.Tv
		self.Tvf_eqn_ref=self.eqn_elem.Tvf

		self.Tv_var_ref=self.var_elem.Tv
		self.Tvf_var_ref=self.var_elem.Tvf
		self.wq=self.lfcnsp_eqn.wqv
		self.wqf=self.lfcnsp_eqn.wqf
		self.eval_elem_eqn_basis=lambda z:self.eval_elem_basis(z,
			                                                   self.eqn.neqn,
			                                                   self.lfcnsp_eqn)
		self.eval_elem_var_basis=lambda z:self.eval_elem_basis(z,
															   self.eqn.nvar,
															   self.eqn.lfcnsp_var)

	def eval_elem_basis(self,z,nc,lfcnsp):
		nv,ndimP1,_=lfcnsp.Qv.shape
		nf=lfcnsp.geom.f2v.shape[1]
		nz=z.shape[1]
		Qv=lfcnsp.eval_basis_vol(z)
		dum=np.zeros([nv,ndimP1,nz,nf])
		elem_basis_=create_elem_basis(nc,Qv,dum)
		Tv=elem_basis_.Tv
		return Tv

#########
#########
class create_elem_strct_mixed2(object):
	"""docstring for create_elem_strct_mixed2"""
	def __init__(self,eqn,neqn1,lfcnsp_eqn1,
		         	  nvar1, lfcnsp_var1,
		         	  neqn2, lfcnsp_eqn2,
		         	  nvar2, lfcnsp_var2):
		self.eqn=eqn
		self.eqn_elem=elem_basis_mixed2(neqn1,
			                            lfcnsp_eqn1.Qvv,
			                            lfcnsp_eqn1.Qvf,
			                            neqn2,
			                            lfcnsp_eqn2.Qvv,
			                            lfcnsp_eqn2.Qvf)
		self.var_elem=elem_basis_mixed2(nvar1,
			                           lfcnsp_var1.Qvv,
			                           lfcnsp_var1.Qvf,
			                           nvar2,
			                           lfcnsp_var2.Qvv,
			                           lfcnsp_var2.Qvf)
		self.Tv_eqn_ref=self.eqn_elem.Tv
		self.Tvf_eqn_ref=self.eqn_elem.Tvf
		self.Tv_var_ref=self.var_elem.Tv
		self.Tvf_var_ref=self.var_elem.Tvf
		self.wq=lfcnsp_eqn1.wqv
		self.wqf=lfcnsp_eqn1.wqf
		self.eval_elem_eqn_basis=lambda z:self.eval_elem_basis(z,
			                                                   neqn1,
			                                                   lfcnsp_eqn1,
			                                                   neqn2,
			                                                   lfcnsp_eqn2)
		self.eval_elem_var_basis=lambda z:self.eval_elem_basis(z,
															   nvar1,
															   lfcnsp_var1,
															   nvar2,
															   lfcnsp_var2)
		


	def eval_elem_basis(self,z,nc1,lfcnsp1,nc2,lfcnsp2):
		[nv1,ndimP1,_,nf]=lfcnsp1.Qvf.shape
		nv2=lfcnsp2.Qvv.shape[0]
		nz=z.shape[1]
		Qv1=lfcnsp1.eval_basis_vol(z)
		Qv2=lfcnsp2.eval_basis_vol(z)
		dum1=np.zeros([nv1,ndimP1,nz,nf])
		dum2=np.zeros([nv2,ndimP1,nz,nf])
		elem_basis=elem_basis_mixed2(nc1,Qv1,dum1,
										nc2,Qv2,dum2)
		return elem_basis.Tv

















				
