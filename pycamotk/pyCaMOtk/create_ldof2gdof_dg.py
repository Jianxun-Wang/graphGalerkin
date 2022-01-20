import numpy as np
import pdb

class create_ldof2gdof_dg(object):
	"""docstring for create_ldof2gdof_cg"""
	def __init__(self,ndof_per_node,nnode_per_elem,nelem):
		self.ndof_per_node=ndof_per_node
		self.nnode_per_elem=nnode_per_elem
		self.nelem=nelem

		self.ndoftotal=self.ndof_per_node*self.nnode_per_elem*self.nelem
		shape=(self.ndof_per_node*self.nnode_per_elem,self.nelem)
		self.ldof2gdof=np.asarray(range(self.ndoftotal)).reshape(shape,order='F')
		
		