import numpy as np
import pdb
from pyCaMOtk.ndist_mltdim import ndist_mltdim_hcube
from pyCaMOtk.tens_core import mltidx_from_linidx
from pyCaMOtk.geom_mltdim import Hypercube
from pyCaMOtk.mesh import Mesh
################################################################################
class mesh_hcube(object):
	"""docstring for mesh_hcube"""
	def __init__(self,etype,lims,nel,porder,**varargin):
		self.etype=etype
		self.lims=lims
		self.nel=np.asarray(nel)
		self.nel=self.nel[:]
		self.porder=porder
		self.varargin=varargin

		self.ndim=np.max(self.nel.shape)
		self.nf=2*self.ndim
		self.nelem=np.prod(self.nel)
		self.coords_sigdim=[]
		for i in range(self.ndim):
			coord_sigmdim_=np.linspace(self.lims[i,0],
				                       self.lims[i,1],
				                       self.nel[i]*self.porder+1)
			self.coords_sigdim.append(coord_sigmdim_)
		self.nnodeperdim=[self.nel[i]*self.porder+1 for i in range(self.ndim)]
		self.nnode=np.prod(np.asarray(self.nnodeperdim))
		self.nnodeperelem=(self.porder+1)**self.ndim
		self.xcg=np.zeros((self.ndim,self.nnode))
		for i in range(self.nnode):
			mltidx=mltidx_from_linidx(self.nnodeperdim,i) #self.nel*self.porder+1
			for j in range(self.ndim):
				self.xcg[j,i]=self.coords_sigdim[j][mltidx[j]]

		if self.ndim==1:
			self.M=range(self.nnode)
		else:
			self.M=np.reshape(range(self.nnode),self.nnodeperdim,order='F');

		self.idx_start=[]
		self.idx_offset=[]
		for k in range(self.ndim):
			self.idx_start.append([i*self.porder for i in range(self.nel[k])])
			self.idx_offset.append([i for i in range(self.porder+1)])

		self.strt=np.zeros(self.nelem)
		self.off=np.zeros(self.nnodeperelem)
		for n in range(self.nelem):
			mltidx1=mltidx_from_linidx(self.nel,n)
			mltidx2=[]
			for d in range(self.ndim):
				mltidx2.append(self.idx_start[d][mltidx1[d]])
			self.strt[n]=self.M[tuple(mltidx2)]
		self.strt=np.sort(self.strt)

		for n in range(self.nnodeperelem):
			mltidx1=mltidx_from_linidx([self.porder+1 for i in range(self.ndim)],n)
			mltidx2=[]
			for d in range(self.ndim):
				#pdb.set_trace()
				mltidx2.append(self.idx_offset[d][(mltidx1[d])])
			self.off[n]=self.M[tuple(mltidx2)]
		self.off=np.sort(self.off)

		self.e2vcg=np.zeros((self.nnodeperelem,self.nelem))
		for e in range(self.nelem):
			self.e2vcg[:,e]=self.strt[e]+self.off
		self.e2vcg=self.e2vcg.astype('int')


		self.e2bnd=np.zeros([2*self.ndim,self.nelem])+np.nan
		self.refhcubeelem=Hypercube(self.ndim,self.porder,'unif')
		self.f2v=self.refhcubeelem.f2n
		for e in range(self.nelem):
			for f in range(self.nf):
				face_nodes=self.xcg[:,self.e2vcg[self.f2v[:,f],e]]
				for d in range(self.ndim):
					if float(np.linalg.norm(face_nodes[d,:]-self.lims[d,0]))==float(0):
						self.e2bnd[f,e]=d
					elif float(np.linalg.norm(face_nodes[d,:]-self.lims[d,1]))==float(0):
						self.e2bnd[f,e]=self.ndim+d
					else:
						pass
		if len(self.varargin)==0:
			self.msh=Mesh(self.etype,self.xcg, self.e2vcg,self.e2bnd)

	def getmsh(self):
		return self.msh
		
		




		




		
#class ClassName(object):
#	"""docstring for ClassName"""
#	def __init__(self, arg):
#		super(ClassName, self).__init__()
#		self.arg = arg
		