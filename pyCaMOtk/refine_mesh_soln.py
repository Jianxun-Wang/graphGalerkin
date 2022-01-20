import numpy as np
import matplotlib.pyplot as plt
import pdb
from pyCaMOtk.lfcnsp import LocalFunctionSpace
from pyCaMOtk.geom_mltdim import create_geom_mltdim_hcube
def refine_mesh_soln(msh0,udg0,max_depth=5,tolx=0.003,tolu=0.012):
	ndim,nv0=msh0.lfcnsp.geom.zk.shape
	nvar=udg0.shape[0]
	nelem=msh0.e2vcg.shape[1]
	neval_per_dim=5 # default is 5
	reffcn=lambda z_,e2bnd_:refine_single_elem(ndim,msh0.etype,z_,e2bnd_)
	if msh0.etype=='hcube':
		lfcnsp_lin=LocalFunctionSpace('Q',ndim,1,'gl',1)
		zeval,_,_,_,_,_=create_geom_mltdim_hcube(ndim,neval_per_dim-1)
	elif msh0.etype=='simp':
		raise ValueError('Simplex element is not supported yet')
	else:
		raise ValueError('Geometry not supported')
	nz=zeval.shape[1]
	nv_lin=lfcnsp_lin.geom.zk.shape[1]
	Qv_lin=lfcnsp_lin.eval_basis_functions(zeval)
	Qv_lin=np.reshape(Qv_lin[:,0,:],[nv_lin,nz],order='F')
	xdg=np.asarray([]) 
	udg=np.asarray([])
	e2bnd=np.asarray([])
	for e in range(nelem):
		xe0=msh0.xcg[:,msh0.e2vcg[:,e]]
		ue0=udg0[:,:,e]
		zk=lfcnsp_lin.geom.zk
		e2bndk=msh0.e2bnd[:,e]
		for k in range(max_depth):
			try:
				nelem_ref=zk.shape[2]
			except:
				nelem_ref=1
			which2ref=[True for i in range(nelem_ref)]

			if not all(which2ref):
				break
			[zk,e2bndk]=refine_multiple_elem(zk,e2bndk,reffcn,which2ref)
		nelem_ref=zk.shape[2]
		zk=zk.reshape([ndim, nv_lin*nelem_ref],order='F')
		Qv_=msh0.lfcnsp.eval_basis_functions(zk)
		Qv_=np.reshape(Qv_[:,0,:],[nv0,nv_lin*nelem_ref],order='F')
		try:
			xdg=np.concatenate((xdg,np.reshape(xe0.dot(Qv_),[ndim,nv_lin,nelem_ref],order='F')),axis=2)
			udg=np.concatenate((udg,np.reshape(ue0.dot(Qv_),[nvar,nv_lin,nelem_ref],order='F')),axis=2)
		except:
			xdg=np.reshape(xe0.dot(Qv_),[ndim,nv_lin,nelem_ref],order='F')
			udg=np.reshape(ue0.dot(Qv_),[nvar,nv_lin,nelem_ref],order='F')
		try:
			e2bnd=np.vstack([e2bnd, e2bndk])
		except:
			e2bnd=e2bndk
	
	return [xdg,udg,e2bnd]
			
		



			
		
			



def refine_single_elem(ndim,etype,zk0,e2bnd0):
	if ndim==1:
		raise ValueError('1-D refine_single_elem is not implemented yet!')
	elif ndim==2:
		if etype=='hcube':
			[zk,e2bnd]=refine_hcube_twodim_single(zk0,e2bnd0)
		elif etype=='simp':
			raise ValueError('Simplex element is not supportted yet!')
		else:
			raise ValueError('Element type is not supported')
	elif ndim==3:
		raise ValueError('3D is not supported')
	else:
		raise ValueError('Dimension is not supported')
	return [zk,e2bnd]

def refine_hcube_twodim_single(zk0,e2bnd0):
	a1=zk0[0,0]
	a2=zk0[1,0]
	b1=zk0[0,3]
	b2=zk0[1,3]
	c1=0.5*(a1+b1)
	c2=0.5*(a2+b2)

	a1=zk0[0,0]
	a2=zk0[1,0]
	b1=zk0[0,3]
	b2=zk0[1,3]
	c1=0.5*(a1+b1)
	c2=0.5*(a2+b2)

	zk=np.zeros([2,4,4])
	zk[:,:,0]=np.asarray([[a1,c1,a1,c1],[a2,a2,c2,c2]])
	zk[:,:,1]=np.asarray([[c1,b1,c1,b1],[a2,a2,c2,c2]])
	zk[:,:,2]=np.asarray([[a1,c1,a1,c1],[c2,c2,b2,b2]])
	zk[:,:,3]=np.asarray([[c1,b1,c1,b1],[c2,c2,b2,b2]])

	e2bnd=np.zeros([4,4])
	e2bnd[:,0]=np.asarray([e2bnd0[0],e2bnd0[1],np.nan,np.nan])
	e2bnd[:,1]=np.asarray([np.nan,e2bnd0[1],e2bnd0[2],np.nan])
	e2bnd[:,2]=np.asarray([e2bnd0[0],np.nan,np.nan,e2bnd0[3]])
	e2bnd[:,3]=np.asarray([np.nan,np.nan,e2bnd0[2],e2bnd0[3]])
	return [zk,e2bnd]

def refine_multiple_elem(zk0,e2bnd0,reffcn,which2ref=None):
	ndim=zk0.shape[0]
	nv=zk0.shape[1]
	if len(zk0.shape)==2:
		nelem = 1
	else:
		nelem=zk0.shape[2]
	if which2ref==None:
		which2ref=[True for i in range(nelem)]
	try:
		[dum,_]=reffcn(zk0[:,:,0],e2bnd0[:,0])
	except:
		[dum,_]=reffcn(zk0[:,:],e2bnd0[:])
	nspawn=dum.shape[2]
	nf=e2bnd0.shape[0]
	zk_=np.zeros([ndim,nv,nspawn,nelem])
	e2bnd_=np.zeros([nf,nspawn,nelem])
	for e in range(nelem):
		try:
			[zk_[:,:,:,e],e2bnd_[:,:,e]]=reffcn(zk0[:,:,e],e2bnd0[:,e])
		except:
			[zk_[:,:,:,e],e2bnd_[:,:,e]]=reffcn(zk0[:,:],e2bnd0[:])
	idx0=0
	nref=len([i for i in which2ref if i is True])
	zk=np.zeros([ndim,nv,nspawn*nref+nelem-nref])
	e2bnd=np.zeros([nf,nspawn*nref+nelem-nref])
	for e in range(nelem):
		if which2ref[e]:
			zk[:,:,idx0:idx0+nspawn]=zk_[:,:,:,e]
			e2bnd[:,idx0:idx0+nspawn]=e2bnd_[:,:,e]
			idx0=idx0+nspawn
		else:
			zk[:,:,idx0]=zk0[:,:,e]
			e2bnd[:,idx0]=e2bnd0[:,e]
			idx0=idx0+1
	return [zk,e2bnd]
		



	













