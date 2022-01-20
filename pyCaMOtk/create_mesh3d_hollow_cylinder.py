from pyCaMOtk.create_mesh_hcube import mesh_hcube
from pyCaMOtk.mesh import Mesh

import numpy as np
import pdb

class create_mesh3d_hollow_cylinder(object):
	"""docstring for create_mesh3d_hollow_cylinder"""
	def __init__(self,c,r1,r2,h,nel,porder):
		self.c=c
		self.r1=r1
		self.r2=r2
		self.h=h
		self.nel=nel
		self.porder=porder

		self.nv1=self.nel[0]*self.porder+1
		self.nv2=self.nel[1]*self.porder+1
		self.nv3=self.nel[2]*self.porder+1

		self.mesh_hcube=mesh_hcube('hcube',
			                       np.asarray([[0,2*np.pi],
			                       	           [self.r1,self.r2],
			                       	           [0,self.h]]),
			                       self.nel,self.porder).getmsh()
		xcg,e2bnd,e2vcg=self.mesh_hcube.xcg,\
		                self.mesh_hcube.e2bnd,\
		                self.mesh_hcube.e2vcg

		self.e2bnd=e2bnd
		idx_temp=np.argwhere(e2bnd==0)
		for i in range(idx_temp.shape[0]):
			self.e2bnd[tuple(idx_temp[i,:])]=np.nan
		
		idx_temp=np.argwhere(e2bnd==1)
		for i in range(idx_temp.shape[0]):
			self.e2bnd[tuple(idx_temp[i,:])]=0

		idx_temp=np.argwhere(e2bnd==2)
		for i in range(idx_temp.shape[0]):
			self.e2bnd[tuple(idx_temp[i,:])]=1

		idx_temp=np.argwhere(e2bnd==3)
		for i in range(idx_temp.shape[0]):
			self.e2bnd[tuple(idx_temp[i,:])]=np.nan

		idx_temp=np.argwhere(e2bnd==4)
		for i in range(idx_temp.shape[0]):
			self.e2bnd[tuple(idx_temp[i,:])]=2

		idx_temp=np.argwhere(e2bnd==5)
		for i in range(idx_temp.shape[0]):
			self.e2bnd[tuple(idx_temp[i,:])]=3

		xcg=np.vstack([xcg[1,:]*np.cos(xcg[0,:])+c[0],
			           xcg[1,:]*np.sin(xcg[0,:])+c[1],
			           xcg[2,:]])

		nv=xcg.shape[1]
		is_periodic=[True,False,False]
		M=np.reshape(range(nv),[self.nv1,self.nv2,self.nv3],order='F')
		idx_rmv=[]; idx_rpl=[]
		idx_rmv1=M[-1,:,:]; idx_rpl1=M[0,:,:]
		idx_rmv2=M[:,-1,:]; idx_rpl2=M[:,0,:]
		idx_rmv3=M[:,:,-1]; idx_rpl3=M[:,:,0]

		if is_periodic[0]:
			idx_rmv.append(idx_rmv1.flatten(order='F'))
			idx_rpl.append(idx_rpl1.flatten(order='F'))

		if is_periodic[1]:
			idx_rmv.append(idx_rmv2.flatten(order='F'))
			idx_rpl.append(idx_rpl2.flatten(order='F'))

		if is_periodic[2]:
			idx_rmv.append(idx_rmv3.flatten(order='F'))
			idx_rpl.append(idx_rpl3.flatten(order='F'))

		idx_rmv=np.hstack(idx_rmv)
		idx_rpl=np.hstack(idx_rpl)

		idx_keep=[i for i in range(nv) if i not in idx_rmv]
		self.xcg=xcg[:,idx_keep]
		nv_rstr=self.xcg.shape[1]
		old2new=np.zeros([nv,1])+np.nan
		old2new[idx_keep]=np.asarray(range(nv_rstr)).reshape(old2new[idx_keep].shape)
		idx_nodes=[i for i in range(nv)]
		for i in range(len(idx_rmv)):
			idx_nodes[idx_rmv[i]]=idx_rpl[i]

		idx_nodes=np.asarray(idx_nodes)
		self.e2vcg=np.squeeze(old2new[idx_nodes[e2vcg]])
		self.msh=Mesh('hcube',self.xcg, self.e2vcg.astype('int'),self.e2bnd)
		
		
	def getmsh(self):
		return self.msh



		