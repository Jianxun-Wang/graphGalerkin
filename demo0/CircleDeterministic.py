import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, DataLoader

from pyCaMOtk.create_mesh_hsphere import mesh_hsphere 
from pyCaMOtk.setup_linelptc_sclr_base_handcode import setup_linelptc_sclr_base_handcode
from pyCaMOtk.create_dbc_strct import create_dbc_strct
from pyCaMOtk.create_femsp_cg import create_femsp_cg
from pyCaMOtk.solve_fem import solve_fem
from pyCaMOtk.visualize_fem import visualize_fem



sys.path.insert(0, '../source')
from FEM_ForwardModel import analyticalPossion, analyticalConeInterpolation
from GCNNModel import e2vcg2connectivity,PossionNet
from TensorFEMCore import Double,solve_fem_GCNN,create_fem_resjac
import setup_prob_eqn_handcode

torch.manual_seed(0)


"""
Hyper prameters
"""
tol=1.0e-16
maxit=500


										
"""
Set up GCNN-FEM Possion problem
"""
nin=1 # Number of input variable
nvar=1 # Number of primanry variable
etype='hcube' # Mesh type
c=[0,0] # Domain center
r=1 # Radius
porder=2 # Polynomial order for solution and geometry basis
nel=[2,2] # Number of element in x and y axis
msh=mesh_hsphere(etype,c,r,nel,porder).getmsh() # Create mesh object
xcg=msh.xcg # Extract node coordinates
ndof=xcg.shape[1]
e2vcg=msh.e2vcg # Extract element connectivity 
connectivity=e2vcg2connectivity(msh.e2vcg,'ele')

"""
e2vcg is a 2D array (NNODE PER ELEM, NELEM): The connectivity of the
mesh. The (:, e)´entries are the global node numbers of the nodes
that comprise element e. The local node numbers of each element are
defined by the columns of this matrix, e.g., e2vcg(i, e) is the
global node number of the ith local node of element e. 
Han Gao currently does not use this information in e2vcg2connectivity function!
"""
bnd2nbc=np.asarray([0]) # Define the boundary tag!
K=lambda x,el: np.asarray([[1],[0],[0],[1]])
"""
# The flux constant Flux=[du/dx, du/dy]^T=K dot [dphi/dx,dphi/dy]
where phi is the solution polynomial function
""" 
Qb=lambda x,n,bnd,el,fc: 0 # The primary variable value on the boundary
dbc_idx=[i for i in range(xcg.shape[1]) if np.sum(xcg[:,i]**2)>1-1e-12] # The boundary node id
dbc_idx=np.asarray(dbc_idx) 
dbc_val=dbc_idx*0 # The boundary node primary variable value
dbc=create_dbc_strct(xcg.shape[1]*nvar,dbc_idx,dbc_val) # Create the class of boundary condition

'''
# Define pameterized problem, in this case, the governing equation is changing and the loss
# function is changing
'''
S=[1] # Parametrize the source value in the pde -F_ij,j=S_i
LossF=[]
for i in S: 
	f=lambda x,el: i
	prob=setup_prob_eqn_handcode.setup_linelptc_sclr_base_handcode(2,K,f,Qb,bnd2nbc)
	femsp=create_femsp_cg(prob,msh,porder,e2vcg,porder,e2vcg,dbc)
	fcn=lambda u_:create_fem_resjac('cg',u_,msh.transfdatacontiguous,
									femsp.elem,femsp.elem_data, 
                   				    femsp.ldof2gdof_eqn.ldof2gdof,
                  			        femsp.ldof2gdof_var.ldof2gdof,
									msh.e2e,femsp.spmat,dbc)
	LossF.append(fcn)

# Define the Training Data
Graph=[]
ii=0
for i in S:
	Ue=Double(analyticalPossion(xcg,i).flatten().reshape(ndof,1))
	fcn_id=Double(np.asarray([ii]))
	Ue_aug=torch.cat((fcn_id,Ue),axis=0)
	Uin=Double(xcg.T)
	graph=Data(x=Uin,y=Ue_aug,edge_index=connectivity)
	Graph.append(graph)
	ii=ii+1
DataList=[[Graph[i]] for i in range(len(S))]
TrainDataloader=DataLoader(DataList,batch_size=1)
# GCNN model
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=PossionNet().to(device)
model=model.double()
# Training Data
[model,info]=solve_fem_GCNN(TrainDataloader,LossF,model,tol,maxit)
print('K=',K)
print('Min Error=',info['Er'].min())
print('Mean Error Last 10 iterations=',np.mean(info['Er'][-10:]))
print('Var  Error Last 10 iterations=',np.var(info['Er'][-10:]))
torch.save(model, './modelCircleDet.pth')

np.savetxt('ErFinal.txt', info['Er'])
np.savetxt('Loss.txt', info['Loss'])

solution=model(Graph[0].to('cuda'))
solution[dbc.dbc_idx]=Double(dbc.dbc_val.reshape([len(dbc.dbc_val),1]))
solution=solution.detach().cpu().numpy()
Ue=Ue.detach().cpu().numpy()

fig=plt.figure()
ax1=plt.subplot(1,2,1)
visualize_fem(ax1,msh,solution[e2vcg],{"plot_elem":True,"nref":6},[])
ax1.set_title('GCNN solution')
ax2=plt.subplot(1,2,2)
visualize_fem(ax2,msh,Ue[e2vcg],{"plot_elem":True,"nref":6},[])
ax2.set_title('Exact solution')
fig.tight_layout(pad=3.0)
plt.savefig('Demo.pdf',bbox_inches='tight')
pdb.set_trace()





