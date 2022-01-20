from __future__ import print_function
import numpy as np
import pdb
from pyCaMOtk.check import is_type
from pyCaMOtk.ndist_mltdim import ndist_mltdim_hcube, ndist_mltdim_simp
from pyCaMOtk.tens_core import linidx_from_mltidx, tensprod_vector_unif,\
							   mltidx_from_linidx

"""
Notation
--------
nvpd : int
  Number of vertices per edge
nvpf : int
  Number of vertices per face
nvpe : int
  Number of vertices per element
nnpd : int
  Number of nodes per edge
nnpf : int
  Number of nodes per face
nnpe : int
  Number of nodes per element
ndpf : int
  Number of edges per face
ndpe : int
  Number of edges per element
nfpe : int
  Number of faces per element

zk : ndarray, shape (ndim, nnpe)
  Coordinates of nodes of reference element
N : ndarray, shape (ndim, nfpe)
  Unit outward normal for each face of reference element
v2n : ndarray, shape (nvpe,)
  Vertex-to-node mapping, i.e., node n coincides with vertex v if n == v2n[v]
f2n : ndarray, shape (nnpf, nfpe)
  Face-to-node mapping, i.e., f2n[:, f] are the nodes on face f
d2n : ndarray, shape (nnpd, ndpe)
  Edge-to-node mapping, i.e., d2n[:, d] are the nodes on edge d
e2n : ndarray, shape (nnpe,)
  Element-to-node mapping, i.e., all node numbers in element (range(nnpe))
"""

def invert_idx_array(nype, x2y):
	"""
	Invert an array of indices (x2y -> y2x), in general, y2x will be a
	list of iterable because of potential for unstructured.

	Input arguments
	---------------
	nype : int
	  Number of "y" objects per element (can only be determined from x2y
	  in special cases)
	x2y : ndarray, shape (nype, nxpe) or (nxpe,) if nype==1
	  Array to invert

	Output argument
	---------------
	y2x : list of iterable, size = nype
	  Inverted array
	"""

	# Extract information from input
	nxpe = x2y.shape[-1]

	# Invert index array
	y2x = []
	for yk in range(nype):
		idx = [xk for xk in range(nxpe) if yk in x2y[:, xk]]
		y2x.append(np.array(idx, dtype=int, order='F'))
	return y2x

def create_geom_mltdim_hcube(ndim, porder, ndist0='unif'):
	"""
	Create geometry and nodes of a high-order hypercube [-1, 1]^ndim; same
	number of nodes and nodal distribution in all dimensions. Reference:
	https://en.wikipedia.org/wiki/Hypercube

	Input arguments
	---------------
	ndim : int
	  Number of dimensions
	porder : int
	  Polynomial degree of completeness
	ndist0 : str
	  Nodal distribution to use in all dimensions

	Output argument
	---------------
	See Notation at top of file
	"""

	# Error/type checking
	if not is_type(ndim, 'int'):
		raise TypeError('ndim must be integral')
	if not is_type(porder, 'int'):
		raise TypeError('porder must be integral')

	# Compute number of geometry objects within hypercube
	nnpd = porder+1
	nnpf = nnpd**(ndim-1)
	nvpe = 2**ndim 
	nnpe = nnpd**ndim
	ndpe = 2**ndim+(ndim-2)*2**(ndim-1)
	nfpe = 2*ndim

	
	# Create nodal distribution within element
	zk = ndist_mltdim_hcube(ndim, nnpd, ndist0)

	## Create all mappings (*-to-node)
	M0 = np.arange(nnpe, dtype=int)
	shp = [nnpd for k in range(ndim)]
	M = M0.reshape(shp, order = 'F')

	# Vertex-to-node mapping
	mltidx = tensprod_vector_unif([0, -1], ndim, flatten=True)
	idx = [linidx_from_mltidx(shp, mltidx[:, k])
												for k in range(mltidx.shape[1])]
	v2n = M0[idx]
	v2n = np.reshape(v2n, (1, nvpe))
	

	# Edge-to-node mapping
	if ndim > 1:
		d2n = np.zeros((nnpd, ndpe), dtype=int, order='F')
		colon = range(nnpd)
		colonEntry = tensprod_vector_unif([0, -1], ndim-1, flatten=True)
		for d in range(ndim):
			#d = ndim - 1 - d # Han Gao add this to sort based on tangent for loop
			idx_into_idx = [j for j in range(ndim) if j!= d]
			for k in range(colonEntry.shape[1]):
				# idx with colon in dth entry and idx0[:, k] in other entries
				idx_linidx = np.zeros(nnpd, order='F', dtype=int)
				idx_mltidx = np.zeros(ndim, order='F', dtype=int)
				for j in range(nnpd):
					idx_mltidx[d] = colon[j]
					idx_mltidx[idx_into_idx] = colonEntry[:, k]
					idx_linidx[j] = linidx_from_mltidx(shp, idx_mltidx)
				d2n[:,d*colonEntry.shape[1] + k] = M0[idx_linidx]
		d2n = d2n.astype(int)
	else:
		d2n = np.zeros((1,2)) # Han Gao add this, ndim == 1, edge becomes point
		d2n[0,0] = 0
		d2n[0,1] = nnpd - 1
		d2n = d2n.astype(int)
	
	# Face-to-node mapping
	if ndim > 1:
		f2n = np.zeros((nnpf, nfpe), dtype=int, order='F')
		faceEntry = tensprod_vector_unif(range(nnpd), ndim-1, flatten=True)
		for d in range(ndim):
			idx_into_idx = [j for j in range(ndim) if j != d]
			idx_linidx0 = np.zeros(nnpf, order='F', dtype=int)
			idx_linidx1 = np.zeros(nnpf, order='F', dtype=int)
			idx_mltidx0 = np.zeros(ndim, order='F', dtype=int)
			idx_mltidx1 = np.zeros(ndim, order='F', dtype=int)
			for k in range(faceEntry.shape[1]):     
				idx_mltidx0[d] = colon[0]
				idx_mltidx1[d] = colon[-1]
				idx_mltidx0[idx_into_idx] = faceEntry[:,k]
				idx_mltidx1[idx_into_idx] = faceEntry[:,k]
				idx_linidx0[k] = linidx_from_mltidx(shp, idx_mltidx0)
				idx_linidx1[k] = linidx_from_mltidx(shp, idx_mltidx1)
			f2n[:, d] = M0[idx_linidx0]
			f2n[:, d+ndim] = M0[idx_linidx1]
	else: 
		f2n = np.zeros((1,2)) # Han Gao add this, ndim == 1, face becomes point
		f2n[0,0] = 0
		f2n[0,1] = nnpd - 1
		f2n = d2n.astype(int)
	

	# Element-to-node mapping
	e2n = np.arange(nnpe, dtype=int)

	# Create outward unit normals based on face numbering
	N = np.zeros((ndim, nfpe), dtype=float, order='F')
	for k in range(ndim):
		N[k, k], N[k, k+ndim] = -1, 1

	return zk, N, v2n, d2n, f2n, e2n



def create_geom_mltdim_simplex(ndim,porder,ndist0):
	nnpd=porder+1;
	nf=ndim+1
	zk=ndist_mltdim_simp(ndim,nnpd,ndist0)
	shp=[nnpd for i in range(ndim)]
	M=np.zeros(shp)+np.nan
	cnt=0; nv_dm1=0
	for i in range(nnpd**ndim):
		idx=mltidx_from_linidx(shp,i)
		if sum(idx)>porder:
			continue
		if sum(idx)==porder:
			nv_dm1=nv_dm1+1
		M[tuple(idx)]=cnt
		cnt=cnt+1
	nnpf=nv_dm1
	M0=M.flatten()

	# face 2 node mapping
	f2n=np.zeros([nv_dm1,nf])
	faceEntry = tensprod_vector_unif(range(nnpd), ndim-1, flatten=True)
	for d in range(ndim):
		idx_into_idx = [j for j in range(ndim) if j != d]
		idx_linidx0 = np.zeros(nnpd**(ndim-1), order='F', dtype=int)
		idx_mltidx0 = np.zeros(ndim, order='F', dtype=int)
		for k in range(faceEntry.shape[1]):     
			idx_mltidx0[d]=0
			idx_mltidx0[idx_into_idx]=faceEntry[:,k]
			idx_linidx0[k] = linidx_from_mltidx(shp, idx_mltidx0)
		start=0
		for i in range(nnpd**(ndim-1)):
			nodeid=M[tuple(mltidx_from_linidx(shp,idx_linidx0[i]))]
			if not np.isnan(nodeid):
				f2n[start,d]=int(nodeid)
				start=start+1
		assert start==nnpf
	contstart=0
	for i in range(nnpd**ndim):
		if sum(mltidx_from_linidx(shp,i))==porder:
			f2n[contstart,-1]=M[tuple(mltidx_from_linidx(shp,i))]
			contstart=1+contstart
	assert contstart==nnpf

	# vortex 2 node mapping
	mltidx_vortex=tensprod_vector_unif([0, porder],ndim,flatten=True).astype('int')
	v2n=[]
	for i in range(mltidx_vortex.shape[1]):
		if np.sum(mltidx_vortex[:,i])<=porder:
			v2n.append(int(M[tuple(mltidx_vortex[:,i])]))
	v2n=np.asarray(v2n).reshape([1,len(v2n)])
	
	# element 2 node mapping
	e2n=np.arange(cnt, dtype=int)

	# edge 2 node mapping
	ndpe=int((ndim+1)*ndim/2)
	if ndim > 1:
		d2n = np.zeros((nnpd, ndpe), dtype=int, order='F')
		colon = range(nnpd)
		colonEntry = tensprod_vector_unif([0, -1], ndim-1, flatten=True)
		start_edge=0
		for d in range(ndim):
			if start_edge>=ndpe:
				break
			#d = ndim - 1 - d # Han Gao add this to sort based on tangent for loop
			idx_into_idx = [j for j in range(ndim) if j!= d]
			for k in range(colonEntry.shape[1]):
				# idx with colon in dth entry and idx0[:, k] in other entries
				idx_linidx = np.zeros(nnpd, order='F', dtype=int)
				idx_mltidx = np.zeros(ndim, order='F', dtype=int)
				for j in range(nnpd):
					idx_mltidx[d] = colon[j]
					idx_mltidx[idx_into_idx] = colonEntry[:, k]
					if np.isnan(M[tuple(idx_mltidx)]):
						break
					idx_linidx[j]=linidx_from_mltidx(shp,idx_mltidx)
				if np.isnan(M[tuple(idx_mltidx)]):
					continue
				d2n[:,start_edge]=M0[idx_linidx]
				start_edge=start_edge+1
		if ndim==2:
			d2n[:,2]=f2n[:,2]
		else:
			for d in range(ndim):
				d2n[:,start_edge]=np.intersect1d(f2n[:,d],f2n[:,ndim])
				start_edge=start_edge+1
		d2n = d2n.astype(int)
	else:
		d2n = np.zeros((1,2)) # Han Gao add this, ndim == 1, edge becomes point
		d2n[0,0] = 0
		d2n[0,1] = nnpd - 1
		d2n = d2n.astype(int)

	# Norm vec
	N = np.zeros((ndim, nf), dtype=float, order='F')
	for k in range(ndim):
		N[k, k]=-1
	N[:,ndim]=1/np.sqrt(ndim)
	return zk, N, v2n, d2n, f2n, e2n



# TODO: Add comments
class Geometry(object):
	def __init__(self, ndim, porder):
		self.ndim = ndim
		self.porder = porder

	def get_nodes_on_face(self, f):
		return self.zk[:, self.f2n[:, f]]

class Hypercube(Geometry):
	def __init__(self, ndim, porder, ndist0='unif'):
		super(Hypercube, self).__init__(ndim, porder)
		zk, N, v2n, d2n, f2n, e2n = create_geom_mltdim_hcube(ndim,
															 porder,
															 ndist0)
		self.zk = zk
		self.N = N
		self.v2n = v2n
		self.d2n = d2n
		self.f2n = f2n
		self.e2n = e2n

class Simplex(Geometry):
	def __init__(self, ndim, porder, ndist0='unif'):
		super(Simplex, self).__init__(ndim, porder)
		zk, N, v2n, d2n, f2n, e2n = create_geom_mltdim_simplex(ndim,
															   porder,
															   ndist0)
		self.zk = zk
		self.N = N
		self.v2n = v2n
		self.d2n = d2n
		self.f2n = f2n
		self.e2n = e2n

if __name__ == '__main__':
	hcube = Hypercube(2, 2)
	simplex=Simplex(3,1)
	pdb.set_trace()
	print('f2n = ', hcube.f2n)
	n2f = invert_idx_array((hcube.porder+1)**(hcube.ndim), hcube.f2n)
	print('n2f = ', n2f)
	#print hcube.f2n[:, 0]
	#zk = hcube(2, 6, 'cheb')
	#import matplotlib.pyplot as plt
	#plt.plot(zk[0, :], zk[1, :], 'bo')
	#plt.show()
