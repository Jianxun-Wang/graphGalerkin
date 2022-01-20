import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pdb
from pyCaMOtk.refine_mesh_soln import refine_mesh_soln
################################################################################
def visualize_fem(ax,msh,udg=[],
	              opts={},
	              which_bnd=[]):
	porder=msh.porder
	[nv,nelem]=msh.e2vcg.shape
	[_,nf]=msh.lfcnsp.geom.f2n.shape
	nsd=msh.nsd
	ndim=msh.ndim
	xcg=msh.xcg
	e2vcg=msh.e2vcg
	e2bnd=msh.e2bnd
	f2v=msh.lfcnsp.geom.f2n
	if udg==[]:
		plot_soln=False
	else:
		plot_soln=True

	if 'nref' in opts:
		nref=opts['nref']
	else:
		nref=0

	if 'climit' in opts:
		colorlimit=opts['climit']
	else:
		colorlimit=None
			

	if 'plot_nodes' in opts:
		plot_nodes=opts['plot_nodes']
	else:
		plot_nodes=False

	if 'plot_elem' in opts:
		plot_elem=opts['plot_elem']
	else:
		plot_elem=False

	nelem_in_udg=len(udg.flatten())
	if udg==[]:
		udg=np.zeros([1,nv,nelem])
	elif nelem_in_udg==nelem:
		udg = np.reshape(np.repmat(udg.flatten(),
			                      [nv,1]),
		                          [1,nv,nelem],
		                          order='F')
	elif nelem_in_udg==int(nv*nelem):
		udg=np.reshape(udg,[1,nv,nelem],order='F')
	else:
		raise ValueError('udg incorrect shape')

	if ndim<3:
		if nref==1 or nref==1.0:
			xdg_ref=xcg[:,msh.e2vcg]
			udg_ref=udg
			[_,nv_lin,nelem_ref]=udg_ref.shape
		else:
			[xdg_ref,udg_ref,_]=refine_mesh_soln(msh, udg, nref)
			[_,nv_lin,nelem_ref]=udg_ref.shape
		udg_ref=np.reshape(udg_ref,[nv_lin,nelem_ref],order='F')
	if ndim==1 and nsd==1:
		raise ValueError('1d plot to be implemented later')
	elif ndim==2 and nsd==2:
		udg0=udg_ref.flatten(order='F')
		xdg0=np.reshape(xdg_ref,[ndim,nv_lin*nelem_ref],order='F')
		e2vcg0=np.reshape(np.asarray(range(nv_lin*nelem_ref)),[nv_lin,nelem_ref],order='F')
		if msh.etype=='hcube':
			idx=np.asarray([0,1,3,2])
		elif msh.etype=='simplex' or msh.etype=='simp':
			idx=[int(i) for i in msh.lfcnsp.geom.v2n[0,:]]#np.asarray([0,1,2])
		else:
			raise ValueError('Only hcube and simplex are implemented by now!')
		if plot_soln:
			patches=[]
			for i in range(nelem_ref):
				polygon_temp=Polygon(xdg0[:,e2vcg0[idx,i]].T,True)
				patches.append(polygon_temp)
			patches_ensemble=PatchCollection(patches,cmap=matplotlib.cm.jet,alpha=1)
			patches_ensemble.set_edgecolor('face')
			patches_ensemble.set_array(np.mean(udg_ref,axis=0))
			#patches_ensemble.set_edgecolors('face')
			#patches_ensemble.set_facecolors()
			#patches_ensemble.set_edgecolors()
			#patches_ensemble.set_snap(True)
			if colorlimit is None:
				pass
			else:
				patches_ensemble.set_clim(colorlimit)
			ax.add_collection(patches_ensemble)
			ax.set_xlim(np.min(xdg0[0,:]),np.max(xdg0[0,:]))
			ax.set_ylim(np.min(xdg0[1,:]),np.max(xdg0[1,:]))
			cbar=plt.colorbar(patches_ensemble,orientation="horizontal")
		if plot_elem:
			if porder==1:
				nr=2
			else:
				nr=20
			r=np.linspace(msh.lfcnsp.rk[0],msh.lfcnsp.rk[-1],nr)
			r=r.reshape([1,len(r)])
			if msh.etype=='simplex' or msh.etype=='simp':
				r=r.T
			Qf=msh.lfcnsp.eval_basis_functions(r)
			if msh.etype=='simplex' or msh.etype=='simp':
				Qf=msh.lfcnsp.Qf
			Qf=np.squeeze(Qf[:,0,:])
			for e in range(nelem):
				for f in range(nf):
					xy=xcg[:,e2vcg[f2v[:,f].astype('int'),e]].dot(Qf)
					if e2bnd[f,e] in which_bnd:
						ax.plot(xy[0,:],xy[1,:],'r-')
					else:
						ax.plot(xy[0,:],xy[1,:],'k-')
		if plot_nodes:
			ax.plot(xcg[0,:],xcg[1,:], 'b.', markersize=20)
	else:
		raise ValueError('Only implemented ndim=2 and nsd=2!')
	#ax.set_xlabel(r'$x$')
	#ax.set_ylabel(r'$y$')
	ax.axis('tight')
	#ax.set_aspect(aspect=1)
	ax.set_aspect('equal')
	return ax,cbar
	'''
	if msh.etype=='hcube':
		idx=np.asarray([0,1,3,2])
	else:
		raise ValueError('Only hcube is implemented by now!')
	if plot_soln:
		patches=[]
		for i in range(nelem_ref):
			polygon_temp=Polygon(xdg0[:,e2vcg0[idx,i]].T,True)
			patches.append(polygon_temp)
		patches_ensemble=PatchCollection(patches,cmap=matplotlib.cm.jet,alpha=1)
		patches_ensemble.set_array(np.mean(udg_ref,axis=0))
		#patches_ensemble.set_facecolors()
		#patches_ensemble.set_edgecolors()
		#patches_ensemble.set_snap(True)
		ax.add_collection(patches_ensemble)
		ax.set_xlim(np.min(xdg0[0,:]),np.max(xdg0[0,:]))
		ax.set_ylim(np.min(xdg0[1,:]),np.max(xdg0[1,:]))
		ax.set_xlabel(r'$x$')
		ax.set_ylabel(r'$y$')
		ax.axis('equal')
		ax.set_aspect(aspect=1)
		plt.colorbar(patches_ensemble)
	if plot_elem:
		if porder==1:
			nr=2
		else:
			nr=20
		r=np.linspace(msh.lfcnsp.rk[0],msh.lfcnsp.rk[-1],nr)
		r=r.reshape([1,len(r)])
		Qf=msh.lfcnsp.eval_basis_functions(r)
		Qf=np.squeeze(Qf[:,0,:])
		for e in range(nelem):
			for f in range(nf):
				xy=xcg[:,e2vcg[f2v[:,f],e]].dot(Qf)
				if e2bnd[f,e] in which_bnd:
					ax.plot(xy[0,:],xy[1,:],'r-',linewidth=2)
				else:
					ax.plot(xy[0,:],xy[1,:],'k-')
	if plot_nodes, plot(ax, xcg(1, :), xcg(2, :), 'b.', 'markersize', 20); end
		
	'''

	

