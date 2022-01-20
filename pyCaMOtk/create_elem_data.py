import numpy as np
import pdb

class create_elem_data(object):
	"""docstring for create_elem_data"""
	def __init__(self,Tv_eqn_ref, Tvf_eqn_ref,
                 Tv_var_ref, Tvf_var_ref,
                 e2e, transf_data, mass_par_fcn=None, 
                 vol_pars_fcn=None,bnd_pars_fcn=None, 
                 face_pars_fcn=None,bnd2nbc=None):

		self.Tv_eqn_ref=Tv_eqn_ref
		self.Tvf_eqn_ref=Tvf_eqn_ref
		self.Tv_var_ref=Tv_var_ref
		self.Tvf_var_ref=Tvf_var_ref
		self.e2e=e2e
		self.transf_data=transf_data
		self.mass_par_fcn=mass_par_fcn
		self.vol_pars_fcn=vol_pars_fcn
		self.bnd_pars_fcn=bnd_pars_fcn
		self.face_pars_fcn=face_pars_fcn
		self.bnd2nbc=bnd2nbc

		"""
		Extract information
		"""
		self.neqn_per_elem,self.neqn,self.ndimP1,self.nq=Tv_eqn_ref.shape
		self.nvar_per_elem,self.nvar,_,_=Tv_var_ref.shape
		self.ndim=self.ndimP1-1
		self.nelem=self.transf_data.e2bnd.shape[1]
		_,self.nqf,self.nf=self.transf_data.xqf.shape[0:3]

		self.nbnd=np.int(np.nanmax(self.transf_data.e2bnd))

		if self.mass_par_fcn==None:
			self.mass_par_fcn=lambda x,el:1
		if self.vol_pars_fcn==None:
			self.vol_pars_fcn=lambda x,el:0
		if self.bnd_pars_fcn==None:
			self.bnd_pars_fcn=lambda x,n,bnd,el,fc:self.vol_pars_fcn(x,el)
		if self.face_pars_fcn==None:
			self.face_pars_fcn=lambda x,n,el1,fc1,el2,fc2:self.vol_pars_fcn(x,el1)
		if len(self.bnd2nbc)==0:
			self.bnd2nbc=np.ones([self.nbnd,1])

		"""
		Parameters: mass matrix, equations, natural boundary conditions
		"""
		self.xtmp=np.zeros([self.ndim,1])
		self.ntmp=np.zeros([self.ndim,1])
		self.m_vol=len(self.vol_pars_fcn(self.xtmp,1))
		self.m_bnd=len(self.bnd_pars_fcn(self.xtmp,self.ntmp,1,1,1))
		self.m_face=len(self.face_pars_fcn(self.xtmp,self.ntmp,1,1,1,1))

		assert self.m_vol==self.m_bnd and self.m_vol==self.m_face,\
		       "Volume, face, and boundary must define same parameter vector"

		self.nbcnbr=np.zeros([self.nf,self.nelem])+np.NaN
		self.mass_par=np.zeros([self.nq,self.nelem])
		self.vol_pars=np.zeros([self.m_vol,self.nq,self.nelem])
		self.bnd_pars=np.zeros([self.m_bnd,self.nqf,self.nf,self.nelem])
		self.face_pars=np.zeros([self.m_face,self.nqf,self.nf,self.nelem])
		self.Tv_eqn_phys=np.zeros([self.neqn_per_elem,self.neqn,self.ndim+1,self.nq,self.nelem])
		self.Tvf_eqn_phys=np.zeros([self.neqn_per_elem,self.neqn,self.ndim+1,self.nqf,self.nf,self.nelem])
		self.Tv_var_phys=np.zeros([self.nvar_per_elem,self.nvar,self.ndim+1,self.nq,self.nelem])
		self.Tvf_var_phys=np.zeros([self.nvar_per_elem,self.nvar,self.ndim+1,self.nqf,self.nf,self.nelem])
		
		for e in range(self.nelem):
			xq=self.transf_data.xq[:,:,e]
			self.Tv_eqn_phys[:,:,0,:,e]=self.Tv_eqn_ref[:,:,0,:]
			self.Tv_var_phys[:,:,0,:,e]=self.Tv_var_ref[:,:,0,:]
			for k in range(self.nq):
				self.mass_par[k,e]=self.mass_par_fcn(xq[:,k],e)
				self.vol_pars[:,k,e]=self.vol_pars_fcn(xq[:,k],e).reshape(self.m_vol,)
				Gi=self.transf_data.Gi[:,:,k,e]
				for j in range(self.neqn):
					self.Tv_eqn_phys[:,j,1:,k,e]=self.Tv_eqn_ref[:,j,1:,k].dot(Gi)
				for j in range(self.nvar):
					self.Tv_var_phys[:,j,1:,k,e]=self.Tv_var_ref[:,j,1:,k].dot(Gi)
			for f in range(self.nf):
				self.Tvf_eqn_phys[:,:,0,:,f,e]=self.Tvf_eqn_ref[:,:,0,:,f]
				self.Tvf_var_phys[:,:,0,:,f,e]=self.Tvf_var_ref[:,:,0,:,f]
				for k in range(self.nqf):
					Gif=self.transf_data.Gif[:,:,k,f,e]
					for j in range(self.neqn):
						self.Tvf_eqn_phys[:,j,1:,k,f,e]=self.Tvf_eqn_ref[:,j,1:,k,f].dot(Gif)
					for j in range(self.nvar):
						self.Tvf_var_phys[:,j,1:,k,f,e]=self.Tvf_var_ref[:,j,1:,k,f].dot(Gif)
				
				bnd=self.transf_data.e2bnd[f,e]
				n=self.transf_data.n[:,:,f,e]
				xqf=self.transf_data.xqf[:,:,f,e]
				
				if np.isnan(bnd):
					ep=self.e2e[f,0,e]
					fp=self.e2e[f,1,e]
					for k in range(self.nqf):
						self.face_pars[:,k,f,e]=self.face_pars_fcn(xqf[:,k],n[:,k],e,f,ep,fp).reshape(self.m_face,)
				else:
					for k in range(self.nqf):
						self.bnd_pars[:,k,f,e]=self.bnd_pars_fcn(xqf[:,k],n[:,k],bnd,e,f).reshape(self.m_bnd,)
				if np.isnan(bnd):
					continue
				self.nbcnbr[f,e]=self.bnd2nbc[self.transf_data.e2bnd[f,e].astype('int')]			
        		#n = transf_data(e).n(:, :, f);
        		#xqf = transf_data(e).xqf(:, :, f);
		






















		