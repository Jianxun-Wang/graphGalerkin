from pyCaMOtk.create_mesh_hcube import mesh_hcube 
import numpy as np
import pdb
from pyCaMOtk.mesh import Mesh
class mesh_hsphere(object):
	"""docstring for mesh_hsphere"""
	def __init__(self,etype,c,r,nel,porder,**varargin):
		#super(mesh_hsphere, self).__init__()
		self.etype=etype
		self.c=np.asarray(c)
		self.r=r
		self.nel=np.asarray(nel)
		self.porder=porder
		self.varargin=varargin

		self.ndim=np.max(self.nel.shape)
		self.dlims=np.zeros([self.ndim,2])
		self.dlims[:,0]=-1;self.dlims[:,1]=1;
		
		self.mesh_hcube=mesh_hcube(self.etype,
			                       self.dlims,
			                       self.nel,
			                       self.porder,
			                       **self.varargin)
		self.xcg0=self.mesh_hcube.xcg
		self.e2vcg=self.mesh_hcube.e2vcg
		self.e2bnd=self.mesh_hcube.e2bnd

		self.nelem=self.e2bnd.shape[1]
		for i in range(self.nelem):
			for j in range(self.ndim*2):
				if not np.isnan(self.e2bnd[j,i]):
					self.e2bnd[j,i]=0 # should be zero or one ask Prof Zahr
		if self.ndim==2:
			self.xcg=np.copy(self.xcg0)
			self.xcg[0,:]=self.xcg[0,:]*np.sqrt(1-self.xcg0[1,:]**2/2)
			self.xcg[1,:]=self.xcg[1,:]*np.sqrt(1-self.xcg0[0,:]**2/2)
		elif self.ndim==3:
			self.xcg=np.copy(self.xcg0)
			self.xcg[0,:]=self.xcg0[0,:]*np.sqrt(1-self.xcg0[1,:]**2/2-self.xcg0[2,:]**2/2+self.xcg0[1,:]**2*self.xcg0[2,:]**2/3)
			self.xcg[1,:]=self.xcg0[1,:]*np.sqrt(1-self.xcg0[2,:]**2/2-self.xcg0[0,:]**2/2+self.xcg0[2,:]**2*self.xcg0[0,:]**2/3)
			self.xcg[2,:]=self.xcg0[2,:]*np.sqrt(1-self.xcg0[0,:]**2/2-self.xcg0[1,:]**2/2+self.xcg0[0,:]**2*self.xcg0[1,:]**2/3)
		else:
			raise ValueError('Dimension not supported')

		self.xcg=self.xcg*self.r
		self.xcg= self.xcg+self.c.reshape(self.ndim,1)

		if len(self.varargin)==0:
			self.msh=Mesh(self.etype,self.xcg, self.e2vcg,self.e2bnd)

	def getmsh(self):
		return self.msh

