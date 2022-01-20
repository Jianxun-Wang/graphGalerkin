import numpy as np
import pdb

class create_dbc_strct(object):
	"""docstring for create_dbc_strct"""
	def __init__(self,ndof,dbc_idx,dbc_val):
		self.ndof=ndof
		self.dbc_idx=np.asarray(dbc_idx)
		self.dbc_val=dbc_val
		self.free_idx=np.setdiff1d(np.asarray(range(self.ndof)),
			                  self.dbc_idx)
		
