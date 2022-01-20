import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, TransformerConv, TAGConv, ARMAConv, SGConv,MFConv, RGCNConv
from torch_geometric.data import InMemoryDataset

import numpy as np
import pdb

def e2vcg2connectivity(e2vcg,type='iso'):
	"""
	e2vcg should be in np.array
	"""
	NnG=np.max(e2vcg)+1
	NnE=e2vcg.shape[1]
	if type=='ele':
		connectivity=[]
		for i in range(NnG):
			positions=np.argwhere(e2vcg==i)[:,0]
			#pdb.set_trace()
			for j in positions:
				for k in range(NnE):
					if e2vcg[j,k]!=i:
						connectivity.append(np.asarray([i,e2vcg[j,k]]))
		return torch.tensor(torch.from_numpy(np.asarray(connectivity).T).to('cuda'),dtype=torch.long)
	elif type=='iso':
		connectivity=[[i for i in range(NnG)],[i for i in range(NnG)]]
		return torch.tensor(torch.from_numpy(np.asarray(connectivity)).to('cuda'),dtype=torch.long)
	elif type=='eletruncate':
		connectivity=[]
		for i in range(NnG):
			positions=np.argwhere(e2vcg==i)[:,0]
			for j in positions:
				for k in range(NnE):
					if e2vcg[j,k]!=i:
						connectivity.append(np.asarray([i,e2vcg[j,k]]))
		return torch.tensor(torch.from_numpy(np.asarray(connectivity).T).to('cuda'),dtype=torch.long)
	



class Ns_Chebnet(torch.nn.Module):
	def __init__(self,split):
		super(Ns_Chebnet, self).__init__()
		nci=2;nco=1
		kk=10
		self.split=split
		self.conv1 = ChebConv(nci, 32,K=kk)
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)

		self.conv11 = ChebConv(nci, 32,K=kk)
		self.conv22 = ChebConv(32, 64,K=kk)
		self.conv33 = ChebConv(64, 128,K=kk)
		self.conv44 = ChebConv(128, 256,K=kk)
		self.conv55 = ChebConv(256, 128,K=kk)
		self.conv66 = ChebConv(128, 64,K=kk)
		self.conv77 = ChebConv(64, 32,K=kk)
		self.conv88 = ChebConv(32, nco,K=kk)


		self.conv111 = ChebConv(nci, 32,K=kk)
		self.conv222 = ChebConv(32, 64,K=kk)
		self.conv333 = ChebConv(64, 128,K=kk)
		self.conv444 = ChebConv(128, 256,K=kk)
		self.conv555 = ChebConv(256, 128,K=kk)
		self.conv666 = ChebConv(128, 64,K=kk)
		self.conv777 = ChebConv(64, 32,K=kk)
		self.conv888 = ChebConv(32, nco,K=kk)
		'''
		torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv8.weight)
		'''
		torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv5.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv6.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv7.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv8.weight)

		torch.nn.init.orthogonal_(self.conv11.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv22.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv33.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv44.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv55.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv66.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv77.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv88.weight)

		torch.nn.init.orthogonal_(self.conv111.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv222.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv333.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv444.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv555.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv666.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv777.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv888.weight)
		#torch.nn.init.orthogonal_(self.conv4.weight)
		#torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		#torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		n1=self.split[0]
		n2=self.split[1]
		n3=self.split[2]
		idx1=[2*i for i in range(n1)]
		idx2=[2*i+1 for i in range(n1)]
		idx3=[i+n1*2 for i in range(n2)]
		x1=x[idx1,:]
		x2=x[idx2,:]
		x3=x[idx3,:]
		edge_index1=edge_index[:,0:n3]
		edge_index2=edge_index[:,n3:2*n3]
		edge_index3=edge_index[:,2*n3:]
		
		x1 = self.conv1(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv2(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv3(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv4(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv5(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv6(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv7(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv8(x1, edge_index1)

		x2 = self.conv11(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv22(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv33(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv44(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv55(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv66(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv77(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv88(x2, edge_index2)

		x3 = self.conv111(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv222(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv333(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv444(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv555(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv666(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv777(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv888(x3, edge_index3)

		uv=[]
		for i in range(n1):
			uv.append(torch.cat([x1[i:i+1,0:],x2[i:i+1,0:]],axis=0))
		uv_=torch.cat(uv,axis=0)
		return torch.cat([uv_,x3],axis=0)#F.log_softmax(x, dim=1)






##############################################
##############################################
# class PossionNet(torch.nn.Module):
# 	def __init__(self,nci=2,nco=1,kk=10):
# 		super(PossionNet, self).__init__()
# 		self.conv1 = ChebConv(nci, 32,K=10)
# 		self.conv2 = ChebConv(32, 64,K=10)
# 		self.conv3 = ChebConv(64, 128,K=10)
# 		self.conv4 = ChebConv(128, 256,K=10)
# 		self.conv5 = ChebConv(256, 128,K=10)
# 		self.conv6 = ChebConv(128, 64,K=10)
# 		self.conv7 = ChebConv(64, 32,K=10)
# 		self.conv8 = ChebConv(32, nco,K=10)
# 		'''
# 		self.conv1 = GATConv(nci, 32,heads=kk)
# 		self.conv2 = GATConv(32, 64,heads=kk)
# 		self.conv3 = GATConv(64, 128,heads=kk)
# 		self.conv4 = GATConv(128, 256,heads=kk)
# 		self.conv5 = GATConv(256, 128,heads=kk)
# 		self.conv6 = GATConv(128, 64,heads=kk)
# 		self.conv7 = GATConv(64, 32,heads=kk)
# 		self.conv8 = GATConv(32, nco,heads=kk)
# 		'''
# 		self.source=torch.tensor([0.25])
# 		self.source =torch.nn.Parameter(self.source)
# 		self.source.requires_grad = True

# 		'''
# 		torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
# 		torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
# 		torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
# 		torch.nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
# 		torch.nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
# 		torch.nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
# 		torch.nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='relu')
# 		torch.nn.init.kaiming_normal_(self.conv8.weight)
# 		'''
# 		torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
# 		torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
# 		torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
# 		torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
# 		torch.nn.init.orthogonal_(self.conv5.weight, torch.nn.init.calculate_gain('relu'))
# 		torch.nn.init.orthogonal_(self.conv6.weight, torch.nn.init.calculate_gain('relu'))
# 		torch.nn.init.orthogonal_(self.conv7.weight, torch.nn.init.calculate_gain('relu'))
# 		torch.nn.init.orthogonal_(self.conv8.weight)
		

# 	def forward(self, data):
# 		#pdb.set_trace()
# 		x, edge_index = data.x, data.edge_index
# 		edge_type = edge_index[1,:] * 0
# 		x = self.conv1(x, edge_index)
# 		x = F.relu(x)
# 		#pdb.set_trace()
# 		x = self.conv2(x, edge_index)
# 		x = F.relu(x)
# 		x = self.conv3(x, edge_index)
# 		x = F.relu(x)
# 		x = self.conv4(x, edge_index)
# 		x = F.relu(x)
# 		x = self.conv5(x, edge_index)
# 		x = F.relu(x)
# 		x = self.conv6(x, edge_index)
# 		x = F.relu(x)
# 		x = self.conv7(x, edge_index)
# 		x = F.relu(x)
# 		x = self.conv8(x, edge_index)
# 		return x#F.log_softmax(x, dim=1)


##############################################
##############################################



class PossionNet(torch.nn.Module):
	def __init__(self,nci=2,nco=1,kk=10):
		super(PossionNet, self).__init__()
		self.conv1 = ChebConv(nci, 32,K=kk)
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)
		'''
		self.conv1 = GATConv(nci, 32,heads=kk)
		self.conv2 = GATConv(32, 64,heads=kk)
		self.conv3 = GATConv(64, 128,heads=kk)
		self.conv4 = GATConv(128, 256,heads=kk)
		self.conv5 = GATConv(256, 128,heads=kk)
		self.conv6 = GATConv(128, 64,heads=kk)
		self.conv7 = GATConv(64, 32,heads=kk)
		self.conv8 = GATConv(32, nco,heads=kk)
		'''
		self.source=torch.tensor([0.25])
		self.source =torch.nn.Parameter(self.source)
		self.source.requires_grad = True

		'''
		torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv8.weight)
		'''
		torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv5.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv6.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv7.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv8.weight)
		#torch.nn.init.orthogonal_(self.conv4.weight)
		#torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		#torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, data):
		#pdb.set_trace()
		x, edge_index = data.x, data.edge_index
		#pdb.set_trace()
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		x = self.conv3(x, edge_index)
		x = F.relu(x)
		x = self.conv4(x, edge_index)
		x = F.relu(x)
		x = self.conv5(x, edge_index)
		x = F.relu(x)
		x = self.conv6(x, edge_index)
		x = F.relu(x)
		x = self.conv7(x, edge_index)
		x = F.relu(x)
		x = self.conv8(x, edge_index)
		return x#F.log_softmax(x, dim=1)



class LinearElasticityNet2D(torch.nn.Module):
	def __init__(self):
		super(LinearElasticityNet2D, self).__init__()
		nci=2;nco=1
		kk=10
		self.conv1 = ChebConv(nci, 32,K=kk)
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)

		self.conv11 = ChebConv(nci, 32,K=kk)
		self.conv22 = ChebConv(32, 64,K=kk)
		self.conv33 = ChebConv(64, 128,K=kk)
		self.conv44 = ChebConv(128, 256,K=kk)
		self.conv55 = ChebConv(256, 128,K=kk)
		self.conv66 = ChebConv(128, 64,K=kk)
		self.conv77 = ChebConv(64, 32,K=kk)
		self.conv88 = ChebConv(32, nco,K=kk)

		self.source=torch.tensor([0.1,0.1])
		self.source =torch.nn.Parameter(self.source)
		self.source.requires_grad = True

	
		'''
		torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv8.weight)
		'''
		torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv5.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv6.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv7.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv8.weight)

		torch.nn.init.orthogonal_(self.conv11.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv22.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv33.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv44.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv55.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv66.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv77.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv88.weight)



	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		n1=int(max(x.shape)/2)
		idx1=[2*i for i in range(n1)]
		idx2=[2*i+1 for i in range(n1)]
		x1=x[idx1,:]
		x2=x[idx2,:]
		edge_index1=edge_index
		edge_index2=edge_index
		
		x1 = self.conv1(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv2(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv3(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv4(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv5(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv6(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv7(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv8(x1, edge_index1)

		x2 = self.conv11(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv22(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv33(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv44(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv55(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv66(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv77(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv88(x2, edge_index2)



		uv=[]
		for i in range(n1):
			uv.append(torch.cat([x1[i:i+1,0:],x2[i:i+1,0:]],axis=0))
		uv_=torch.cat(uv,axis=0)
		#pdb.set_trace()
		return uv_#F.log_softmax(x, dim=1)	




class ThreeDElasticityNet_Cheb(torch.nn.Module):
	def __init__(self):
		super(ThreeDElasticityNet_Cheb, self).__init__()
		nci=3;nco=1
		kk=10
		self.conv1 = ChebConv(nci, 32,K=kk)
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)

		self.conv11 = ChebConv(nci, 32,K=kk)
		self.conv22 = ChebConv(32, 64,K=kk)
		self.conv33 = ChebConv(64, 128,K=kk)
		self.conv44 = ChebConv(128, 256,K=kk)
		self.conv55 = ChebConv(256, 128,K=kk)
		self.conv66 = ChebConv(128, 64,K=kk)
		self.conv77 = ChebConv(64, 32,K=kk)
		self.conv88 = ChebConv(32, nco,K=kk)


		self.conv111 = ChebConv(nci, 32,K=kk)
		self.conv222 = ChebConv(32, 64,K=kk)
		self.conv333 = ChebConv(64, 128,K=kk)
		self.conv444 = ChebConv(128, 256,K=kk)
		self.conv555 = ChebConv(256, 128,K=kk)
		self.conv666 = ChebConv(128, 64,K=kk)
		self.conv777 = ChebConv(64, 32,K=kk)
		self.conv888 = ChebConv(32, nco,K=kk)
		'''
		torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv8.weight)
		'''
		torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv5.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv6.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv7.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv8.weight)

		torch.nn.init.orthogonal_(self.conv11.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv22.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv33.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv44.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv55.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv66.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv77.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv88.weight)

		torch.nn.init.orthogonal_(self.conv111.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv222.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv333.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv444.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv555.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv666.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv777.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.conv888.weight)
		#torch.nn.init.orthogonal_(self.conv4.weight)
		#torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		#torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		#pdb.set_trace()
		n1=int(max(x.shape)/3)
		idx1=[3*i for i in range(n1)]
		idx2=[3*i+1 for i in range(n1)]
		idx3=[3*i+2 for i in range(n1)]
		x1=x[idx1,:]
		x2=x[idx2,:]
		x3=x[idx3,:]
		edge_index1=edge_index
		edge_index2=edge_index
		edge_index3=edge_index
		
		x1 = self.conv1(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv2(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv3(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv4(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv5(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv6(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv7(x1, edge_index1)
		x1 = F.relu(x1)
		x1 = self.conv8(x1, edge_index1)

		x2 = self.conv11(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv22(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv33(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv44(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv55(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv66(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv77(x2, edge_index2)
		x2 = F.relu(x2)
		x2 = self.conv88(x2, edge_index2)

		x3 = self.conv111(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv222(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv333(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv444(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv555(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv666(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv777(x3, edge_index3)
		x3 = F.relu(x3)
		x3 = self.conv888(x3, edge_index3)

		uv=[]
		for i in range(n1):
			uv.append(torch.cat([x1[i:i+1,0:],x2[i:i+1,0:],x3[i:i+1,0:]],axis=0))
		uv_=torch.cat(uv,axis=0)
		#pdb.set_trace()
		return uv_#F.log_softmax(x, dim=1)	


	
