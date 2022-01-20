from pyCaMOtk.lfcnsp import LocalFunctionSpace
from pyCaMOtk.transf import ElementTransformation
import pdb
import numpy as np
from pyCaMOtk.qrule_onedim import * 
from pyCaMOtk.poly_onedim import *
try:
    from scipy.misc import comb
except:
    from scipy.special import comb
################################################################################
# xcg : nsd x nv : nodes
# e2vcg : nnpe x ne : connectivity
# e2bnd : nfpe x ne : boundary tags


class Mesh(object):
    def __init__(self, etype, xcg, e2vcg, e2bnd, ndim=None,
                 qrule='gl', nq0=None):
        if ndim is None: ndim = xcg.shape[0]
        nvpe=e2vcg.shape[0]
        self.etype=etype
        self.xcg=xcg.copy('F')
        self.e2vcg=e2vcg.copy('F')
        self.e2bnd=e2bnd.copy('F')
        self.ndim=ndim
        self.nsd=xcg.shape[0]
        if etype == 'hcube':
            if ((float(nvpe)**(1.0/ndim)-1)-int(float(nvpe)**(1.0/ndim)-1))>0.9:
                self.porder=int(float(nvpe)**(1.0/ndim)-1)+1
            else:
                self.porder=int(float(nvpe)**(1.0/ndim)-1)
            self.LFtype='Q'
        elif etype=='simp' or etype=='simplex':
            self.LFtype='P'
            nv0=0; porder=0
            while True:
                nv0=nv0+comb(porder+ndim-1,ndim-1)
                if nv0==nvpe:
                    break
                porder=porder+1
            self.porder=porder
        else               : raise ValueError('etype not supported')
        if nq0==None or nq0==[]:
            self.nq0=max(3*self.porder,1)
        else:
            self.nq0=nq0
        self.lfcnsp=LocalFunctionSpace(self.LFtype, self.nsd, self.porder)
        self.transf_data=ElementTransformation()
        if self.ndim==self.nsd:
            self.e2e=self.setup_mesh_elem2elem_conn(self.e2vcg,self.lfcnsp.geom.f2n)
            [self.transfdatanoncontiguous, 
             self.transfdatacontiguous]=self.create_transf_data_ndim(self.lfcnsp,self.xcg,
                                                                     self.e2vcg, self.e2bnd,
                                                                     self.e2e)
        elif self.nsd>1 and self.ndim==1:
            raise ValueError('for nsd>1 and ndim=1, we do not implement yet!')
            pass


    def setup_mesh_elem2elem_conn(self, e2vcg, f2v):
        nelem=self.e2vcg.shape[1]
        nvf=f2v.shape[0]
        nf=f2v.shape[1]
        f2cg=np.zeros((nvf, nelem, nf));
        for i in range(nf):
            f2cg[:, :, i]=e2vcg[f2v[:, i].astype(int), :]
        f2cg=np.reshape(f2cg, (nvf, nelem*nf), order = 'F')
        dummy,JC,IC= np.unique(np.transpose(np.sort(f2cg, axis=0)), 
                               return_index=True, return_inverse=True, 
                               return_counts=False, axis=0)
        f2cg = f2cg[:, JC]
        e2f=np.transpose(np.reshape(IC, (nelem,nf), order='F'))
        ne_e2f = e2f.shape[0]*e2f.shape[1]
        e2f_s2e=np.reshape(e2f, (ne_e2f,1),order='F')
        e2f_e2s=[e2f_s2e[ne_e2f-1-i] for i in range(ne_e2f)]
        dummy,I1=np.unique(e2f_s2e,return_index=True)
        dummy,I2=np.unique(e2f_e2s,return_index=True)
        I2=ne_e2f-1-I2
        E1=np.floor((I1)/nf)
        F1=I1-nf*E1
        E2=np.floor((I2)/nf) 
        F2=I2-nf*E2
        f2e_elem=np.transpose(np.vstack((E1,E2))).astype('int')
        f2e_face=np.transpose(np.vstack((F1,F2))).astype('int')
        nfg=f2cg.shape[1]
        e2e=np.zeros((nf,2,nelem))+np.nan
        for fg in range(nfg):
            e1 = f2e_elem[fg, 0]
            e2 = f2e_elem[fg, 1]
            if e1==e2:
                continue
            f1 = f2e_face[fg, 0]
            f2 = f2e_face[fg, 1]
            e2e[f1, 0, e1] = e2
            e2e[f1, 1, e1] = f2
            e2e[f2, 0, e2] = e1
            e2e[f2, 1, e2] = f1
        return e2e

    def create_transf_data_ndim(self,geom,xcg,e2vcg,e2bnd,e2e):
        # Extract relevant information (Here geom means LFspace)
        ndim=xcg.shape[0]
        nnode_per_elem,nelem=e2vcg.shape
        nx=geom.Qvv.shape[2]
        nxf=geom.Qvf.shape[2]
        nf=geom.geom.f2n.shape[1]

        # Compute isoparametric quantities for each element
        xe=np.zeros((ndim, nnode_per_elem, nelem))


        xq=np.zeros((ndim, nx, nelem))
        xqf=np.zeros((ndim, nxf, nf, nelem))
        detG=np.zeros((nx, nelem))
        sigf=np.zeros((nxf, nf, nelem))
        Gi=np.zeros((ndim, ndim, nx, nelem))
        Gif=np.zeros((ndim, ndim, nxf, nf, nelem))
        n=np.zeros((ndim, nxf, nf, nelem))
        transfdatanoncontiguous=[]
        for e in range(nelem):
            xe_=xcg[:,e2vcg[:,e]]
            self.transf_data.eval_transf_quant(xe_, geom)
            xq_=self.transf_data.xq
            detG_=self.transf_data.detG
            Gi_=self.transf_data.Gi
            xqf_=self.transf_data.xqf
            sigf_=self.transf_data.sigf
            Gif_=self.transf_data.Gif
            n_ =self.transf_data.n
            transfdatanoncontiguous.append(transf_data_elem(xe_,
                                                            xq_,
                                                            detG_,
                                                            Gi_,
                                                            xqf_,
                                                            sigf_,
                                                            Gif_,
                                                            n_))
            xe[:,:,e]=xe_
            xq[:,:,e]=xq_
            detG[:,e]=detG_.flatten(order='F')
            Gi[:,:,:,e]=Gi_
            xqf[:,:,:,e]=xqf_
            sigf[:,:,e]=sigf_
            Gif[:,:,:,:,e]=Gif_
            n[:,:,:,e]=n_
        perm=self.align_face_quad_with_neighbors(e2e,xqf)
        #pdb.set_trace()
        transfdatacontiguous=transf_data_contiguous(np.squeeze(e2bnd), 
                                                    np.squeeze(xe), 
                                                    np.squeeze(xq), 
                                                    np.squeeze(xqf),
                                                    np.squeeze(detG), 
                                                    np.squeeze(Gi), 
                                                    np.squeeze(sigf), 
                                                    np.squeeze(Gif),
                                                    np.squeeze(n), 
                                                    np.squeeze(perm))
        return transfdatanoncontiguous, transfdatacontiguous
       

    def align_face_quad_with_neighbors(self, e2e, xqf):
        # Extract relevant information 
        [nf,trash,nelem]=e2e.shape
        nqf=xqf.shape[1]
        d=np.zeros((nqf, 1))
        perm=np.zeros((nqf, nf, nelem))+np.nan
        for e in range(nelem):
            for f in range(nf):
                ep = e2e[f, 0, e]
                fp = e2e[f, 1, e]
                if np.isnan(ep):
                    continue 
                ep=int(ep)
                fp=int(fp)
                xqfI=xqf[:, :, f, e]
                xqfO=xqf[:, :, fp, ep]
                for k0 in range(nqf):
                    for k1 in range(nqf):
                        d[k1]=np.linalg.norm(xqfI[:, k0]-xqfO[:, k1])
                    perm[k0,f,e]=np.argmin(d)
        return perm

    def getelemtransfdata(self,e):
        return self.transfdatanoncontiguous[e]

    '''
    def create_transf_data_onedim(self,geom,xcg,e2vcg):
        qrule0='gl'
        ndim=xcg.shape[0]
        nnode_per_elem,nelem=e2vcg.shape
        [nv,garbage,nx]=geom.Qvv.shape
        wqd, sq=qrule_onedim(4*(nv-1), qrule0)
        xe=np.zeros((ndim,nnode_per_elem,nelem))
        sq=np.zeros((nx, nelem))
        xq=np.zeros((ndim, nx, nelem))
        xinvq=np.zeros((nx, nelem))
        for e in range(nelem):
            xe_=xcg[:,e2vcg[:,e]]
            xe[:,:,e]=xe_
        # not done yet

    def eval_arclen_map(self,z,xe,wq=None,zq=None):
        nv=xe.shape[1]
        if wq is None and qz is None:
            [wq,zq] = create_quad_onedim_gaussleg(4*nv)
        zk=np.linspace(-1, 1, nv)
        Q=eval_poly_nodal_onedim(zk, 0.5*(1+z)*(zq+1)-1)
        dQ0=np.squeeze(Q[:, 1, :])
        lq=np.sqrt(np.sum((xe.dot(dQ0))**2, 1))*0.5*(1+z)
        l = wq(:)*lq(:)
        # not done yet need talk to Prof. Zahr
    '''


        


class transf_data_elem(object):
    """docstring for transfdataelem"""
    def __init__(self,xe,xq,detG,Gi,xqf,sigf,Gif,n):
        self.xe=xe
        self.xq=xq
        self.detG=detG
        self.Gi=Gi
        self.xqf=xqf
        self.Gif=Gif
        self.n=n

class transf_data_contiguous(object):
    def __init__(self,e2bnd,xe,xq,xqf,detG,Gi,sigf,Gif,n,perm):
        self.e2bnd=e2bnd
        self.xe=xe
        self.xq=xq
        self.xqf=xqf
        self.detG=detG
        self.Gi=Gi
        self.sigf=sigf
        self.Gif=Gif
        self.n=n
        self.perm=perm


        
        
        
################################################################################
def get_gdof_from_bndtag(ldof,bndtag,ndof_per_node,ldof2gdof,e2bnd,f2v):
    # Extract information from input
    f2v=f2v.astype('int')
    nf=f2v.shape[1]
    nelem=ldof2gdof.shape[1]
    nnode_per_elem=int(ldof2gdof.shape[0]/ndof_per_node)
    ldof2gdof=np.reshape(ldof2gdof,[ndof_per_node,nnode_per_elem,nelem],order='F')
    gdof_idx = []
    for e in range(nelem):
        for f in range(nf):
            if np.isnan(e2bnd[f,e]):
                continue
            if int(e2bnd[f,e]) not in bndtag:
                continue
            for i in ldof:
                for j in f2v[:,f]:
                    if ldof2gdof[i,j,e] not in gdof_idx:
                        gdof_idx.append(ldof2gdof[i,j,e])                    
    return gdof_idx




        

       


        
