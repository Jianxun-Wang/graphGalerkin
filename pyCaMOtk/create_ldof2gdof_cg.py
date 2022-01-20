import numpy as np
import numpy.matlib
import pdb
################################################################################
class create_ldof2gdof_cg(object):
	"""docstring for create_ldof2gdof_cg"""
	def __init__(self, ndof_per_node, e2vcg):
		self.ndof_per_node=ndof_per_node
		self.e2vcg=e2vcg
		# Extract info
		self.nnode_per_elem=e2vcg.shape[0]
		self.nelem=e2vcg.shape[1]
		# Initialize ldof2gdof
		self.ldof2gdof=np.zeros((self.ndof_per_node*self.nnode_per_elem,
			                     self.nelem))
		for e in range(self.nelem):
			gdof=np.matlib.repmat(self.e2vcg[:, e].T*self.ndof_per_node,
				                  self.ndof_per_node,1)+\
				 np.asarray(range(self.ndof_per_node)).reshape(-1,1)
			self.ldof2gdof[:, e]=gdof.reshape(self.ndof_per_node*self.nnode_per_elem,order='F')
		self.ldof2gdof=self.ldof2gdof.astype('int')


class create_ldof2gdof_cg_mixed2(object):
	"""docstring for create_ldof2gdof_cg_mixed2"""
	def __init__(self,ndof_per_node1,e2vcg1,
		              ndof_per_node2,e2vcg2):
		self.ndof_per_node1=ndof_per_node1
		self.e2vcg1=e2vcg1
		self.ndof_per_node2=ndof_per_node2
		self.e2vcg2=e2vcg2

		self.nnode_per_elem1=self.e2vcg1.shape[0]
		self.nelem1=self.e2vcg1.shape[1]
		self.nnode_per_elem2=self.e2vcg2.shape[0]
		self.nelem2=self.e2vcg2.shape[1]

		assert self.nelem1==self.nelem2,"Meshes must have same number of elements!"
		self.nelem=self.nelem1

		# Preallocate map from local to global degrees of freedom
		self.nnode1=np.max(self.e2vcg1)
		self.ldof2gdof=np.zeros((self.ndof_per_node1*self.nnode_per_elem1+\
                  				 self.ndof_per_node2*self.nnode_per_elem2,self.nelem))
		
		for e in range(self.nelem):
			gdof1=np.matlib.repmat(self.e2vcg1[:, e].T*self.ndof_per_node1,
				                  self.ndof_per_node1,1)+\
				 np.asarray(range(self.ndof_per_node1)).reshape(-1,1)

			gdof2=np.matlib.repmat(self.e2vcg2[:, e].T*self.ndof_per_node2,
				                  self.ndof_per_node2,1)+\
				 np.asarray(range(self.ndof_per_node2)).reshape(-1,1)
			gdof1=gdof1.reshape(-1,1,order='F')
			gdof2=gdof2.reshape(-1,1,order='F')+self.ndof_per_node1*(self.nnode1+1)
			gdof=np.vstack((gdof1, gdof2))
			self.ldof2gdof[:, e]=gdof.reshape(len(gdof),order='F')
		self.ldof2gdof=self.ldof2gdof.astype('int')





    
		