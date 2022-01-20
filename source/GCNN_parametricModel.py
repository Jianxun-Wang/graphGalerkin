import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, SplineConv
from torch_geometric.data import InMemoryDataset

import numpy as np
import pdb











class PossionNet_Parametric_Cheb(torch.nn.Module):
	def __init__(self,nci=2,nco=1,kk=10):
		super(PossionNet_Parametric_Cheb, self).__init__()
		self.conv1 = ChebConv(nci, 32,K=kk)
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)

		self.convA = ChebConv(nco, 32,K=kk)
		self.convB = ChebConv(32, 32,K=kk)
		self.convC = ChebConv(32, nco,K=kk)
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

		torch.nn.init.orthogonal_(self.convA.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.convB.weight, torch.nn.init.calculate_gain('relu'))
		torch.nn.init.orthogonal_(self.convC.weight)
		#torch.nn.init.orthogonal_(self.conv4.weight)
		#torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		#torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
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

		x = self.convA(x, edge_index)
		x = F.relu(x)

		x = self.convB(x, edge_index)
		x = F.relu(x)

		x = self.convC(x, edge_index)
		


		return x#F.log_softmax(x, dim=1)