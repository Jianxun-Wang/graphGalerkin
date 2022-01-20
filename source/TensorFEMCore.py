import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import sparse
import torch
import time
import os
################################################################################

def eval_unassembled_resjac_claw_cg(U,transf_data,elem,
                                    elem_data,ldof2gdof_var,parsfuncI,parsfuncB):
	"""Evaluate elementwise residual and jacobian of conservation law in CG"""
	nelem=elem_data.nelem
	neqn_per_elem=elem.Tv_eqn_ref.shape[0]
	nvar_per_elem=elem.Tv_var_ref.shape[0]
	Re=np.zeros([neqn_per_elem,nelem])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem,nelem])
	Re=Double(Re)
	dRe=Double(dRe)
	Re_=[]
	dRe_=[]
	for e in range(nelem):
		Ue=U[ldof2gdof_var[:,e]]
		Re0_,dRe0_=intg_elem_claw_vol(Ue,transf_data,elem,elem_data,e,parsfuncI)
		Re1_,dRe1_=intg_elem_claw_extface(Ue,transf_data,elem,elem_data,e,parsfuncB)
		Re_.append(ReshapeFix((Re0_+Re1_),[neqn_per_elem,1],order='F'))
		dRe_.append(ReshapeFix((dRe0_+dRe1_),[neqn_per_elem,nvar_per_elem,1],order='C'))  
	Re=torch.cat(Re_,axis=1)
	dRe=torch.cat(dRe_,axis=2)
	return Re,dRe
		

def create_fem_resjac(fespc,Uf,transf_data,elem,elem_data,
                      ldof2gdof_eqn,ldof2gdof_var,e2e,spmat,dbc,enforce_idx=None,parsfuncI=None,parsfuncB=None):
	"""Create global residual(loss) and jacobian of conservation law in CG"""
	ndof_var=np.max(ldof2gdof_var[:])+1
	dbc_idx=dbc.dbc_idx
	dbc_val=dbc.dbc_val
	free_idx=dbc.free_idx
	# Uf is the GCNN output
	Uf=ReshapeFix(Uf,[ndof_var,1],'C')
	U_temp=torch.zeros([ndof_var,1]).double().to('cuda')
	U_temp[dbc_idx,0:]=torch.from_numpy(dbc_val).to('cuda').double().reshape(len(dbc_val),1)\
		          -Uf[dbc_idx,0:]
	U=U_temp+Uf
	# U is the GCNN output hardimpose BC but can backPP
	if fespc=='cg' or fespc=='CG':
		Re,dRe=eval_unassembled_resjac_claw_cg(U,transf_data,elem,elem_data,
			                                   ldof2gdof_var,parsfuncI,parsfuncB)
		dR=assemble_nobc_mat(dRe,spmat.cooidx,spmat.lmat2gmat)
	else:
		raise ValueError('FE space only support cg!')
	R=assemble_nobc_vec(Re,ldof2gdof_eqn)
	if enforce_idx==None:
		Rf=R[free_idx]
	else:
		Rf=R[enforce_idx]
	dRf=dR.tocsr()[free_idx,:]
	dRf=dRf.tocsr()[:,free_idx].T
	print('Max Rf ===============================',torch.max(torch.abs(Rf)))
	return Rf,dRf,dbc
	
def intg_elem_claw_vol(Ue,transf_data,elem,elem_data,e,parsfuncI=None):
	"""Intergrate elementwise internal volume of element residual and jacobian of conservation law"""
	[neqn_per_elem,neqn,ndimP1,nq]=elem.Tv_eqn_ref.shape
	[nvar_per_elem,nvar,_,_]=elem.Tv_var_ref.shape
	ndim=ndimP1-1
	wq=elem.wq
	detG=transf_data.detG[:,e]
	Tvar=elem_data.Tv_var_phys[:,:,:,:,e].reshape([nvar_per_elem,nvar*(ndim+1)*nq],
												  order='F')
	Re=np.zeros([neqn_per_elem,1])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem])
	Tvar_tensor=torch.from_numpy(Tvar).double().to('cuda')
	UQq=ReshapeFix(torch.mm(Tvar_tensor.T,Ue),[nvar,ndim+1,nq],'F')
	w=wq*detG
	Re=Double(Re)
	dRe=Double(dRe)
	for k in range(nq):
		Teqn=elem_data.Tv_eqn_phys[:,:,:,k,e].reshape([neqn_per_elem,neqn*(ndim+1)],
			                                      order='F')
		Tvar=elem_data.Tv_var_phys[:,:,:,k,e].reshape([nvar_per_elem,nvar*(ndim+1)],
			                                      order='F')
		x=transf_data.xq[:,k,e]
		if parsfuncI==None:
			pars=elem_data.vol_pars[:,k,e]
		else:
			pars=parsfuncI(x)
		SF,dSFdU=elem.eqn.srcflux(UQq[:,:,k],pars,x)
		dSFdU=ReshapeFix(dSFdU,[neqn*(ndim+1),nvar*(ndim+1)],order='F')
		Teqn=Double(Teqn)
		Tvar=Double(Tvar)
		#SF=SF.flatten()
		SF=ReshapeFix(SF,[len(SF.flatten()),1])
		Re=Re-w[k]*ReshapeFix(torch.mm(Teqn,SF),Re.shape,order='F')
		dRe=dRe-w[k]*torch.mm(Teqn,torch.mm(dSFdU,Tvar.T))
	return Re, dRe
		
def intg_elem_claw_extface(Ue,transf_data,elem,elem_data,e,parsfuncB=None):
	"""Intergrate elementwise the boundary face of element residual and jacobian of conservation law """
	[neqn_per_elem,neqn,ndimP1,nqf,nf]=elem.Tvf_eqn_ref.shape
	[nvar_per_elem,nvar,_,_,_]=elem.Tvf_var_ref.shape
	ndim=ndimP1-1
	wqf=elem.wqf
	sigf=transf_data.sigf[:,:,e]
	nbcnbr=elem_data.nbcnbr[:,e]
	Re=np.zeros([neqn_per_elem,1])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem])
	wf=wqf[:].reshape(len(wqf),1)*sigf
	Re=Double(Re)
	dRe=Double(dRe)
	for f in range(nf):
		if np.isnan(nbcnbr[f]):
			continue
		Tvar=np.reshape(elem_data.Tvf_var_phys[:,:,:,:,f,e],
						[nvar_per_elem,nvar*(ndim+1)*nqf],order='F')
		Tvar=Double(Tvar)
		UQqf=ReshapeFix(torch.mm(Tvar.T,Ue),[nvar,ndim+1,nqf],order='F')
		for k in range(nqf):
			x=transf_data.xqf[:,k,f,e]
			n=transf_data.n[:,k,f,e]
			Teqn=elem_data.Tvf_eqn_phys[:,:,0,k,f,e]
			Tvar=np.reshape(elem_data.Tvf_var_phys[:,:,:,k,f,e],
				            [nvar_per_elem,nvar*(ndim+1)],order='F')
			Teqn=Double(Teqn)
			Tvar=Double(Tvar)
			if parsfuncB==None:
				pars=elem_data.bnd_pars[:,k,f,e]
			else:
				pars=parsfuncB(x)
			_,_,Fb,dFbdU=elem.eqn.bndstvcflux(nbcnbr[f],UQqf[:,:,k],pars,x,n)
			dFbdU=ReshapeFix(dFbdU,[neqn,nvar*(ndim+1)],order='F')
			Re=Re+wf[k,f]*torch.mm(Teqn,Fb)
			dRe=dRe+wf[k,f]*torch.mm(Teqn,torch.mm(dFbdU,Tvar.T))
	return Re,dRe
			
def assemble_nobc_mat(Me,cooidx,lmat2gmat):
	"""Assembly global jacobian of conservation law (currently no use)"""
	Me=Me.detach().cpu().numpy()
	nnz=cooidx.shape[0]
	cooidx=cooidx.astype('int')
	Mval=np.zeros(nnz)
	nelem=Me.shape[2]
	for e in range(nelem):
		idx=lmat2gmat[:,:,e]
		Mval[idx]=Mval[idx]+Me[:,:,e]
	M=sparse.coo_matrix((Mval,(cooidx[:,0],cooidx[:,1])))
	return M

def assemble_nobc_vec(Fe,ldof2gdof_eqn):
	"""Assembly global residual of conservation law (!!very useful!!)"""
	ndof=np.max(ldof2gdof_eqn[:])+1
	nelem=Fe.shape[1]
	F=np.zeros(ndof)
	F=Double(F)
	for e in range(nelem):
		idx=ldof2gdof_eqn[:,e]
		F[idx]=F[idx]+Fe[:,e]	
	return F
'''
def solve_fem(fespc,transf_data,elem, 
			  elem_data,ldof2gdof_eqn, 
			  ldof2gdof_var, e2e, spmat, 
			  dbc,DataLoader,model,tol=1e-3,maxit=2000):
	"""Wrapper""" 
	ndof_var=np.max(ldof2gdof_var[:])+1
	dbc_idx=dbc.dbc_idx
	dbc_val=dbc.dbc_val
	free_idx=dbc.free_idx

	fcn=lambda u_:create_fem_resjac(fespc,u_,transf_data,
	                                elem,elem_data,ldof2gdof_eqn,
									ldof2gdof_var,e2e,spmat,dbc)
	model,info=solve_SGD(DataLoader,fcn,model,tol,maxit)
	return model, info
'''
def solve_fem_GCNN(DataLoader,LossF,model,tol=1e-3,maxit=2000,qoiidx=None,softidx=None,penaltyConstant=None):
	"""Wrapper""" 
	startime=time.time()
	model,info=solve_SGD(DataLoader,LossF,model,tol,maxit,qoiidx,softidx,penaltyConstant)
	print('wallclock time of all epochs = ',time.time()-startime)
	return model, info



def solve_SGD(DataLoader,LossF,model,tol,maxit,qoiidx,softidx,penaltyConstant,plotFlag='True'):
	"""
	DataLoader: training data
	fcn: loss function
	model: GCNN model to be trained
	tol: the trauncation of loss function
	maxit: the maximum number of epoch
	"""
	optimizer=torch.optim.Adam(model.parameters(), lr=0.001) #model.parameters() default 0.001
	criterion=torch.nn.MSELoss()
	Er=[];	Loss=[]
	tol_e=[1,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,
	       0.005,0.001,0.0009,0.0008,0.0007,
		   0.0006,0.0005,0.0004,0.0003,0.0002,
		   0.0001,0.00001]
	idx_tol_e=0
	for epoch in range(maxit):
		print('epoch = ',epoch)
		startime=time.time()
		er,loss,model=trainmodel(DataLoader,LossF,model,
		           				 optimizer,criterion,qoiidx,softidx,penaltyConstant)
		print('Solution er = ',er)
		print('wallclock time of this epoch= ',time.time()-startime)
		Er.append(er);Loss.append(loss)
		if loss<tol or er<tol_e[idx_tol_e]:
			idx_tol_e=idx_tol_e+1
			print('The training reaches the expected loss!')
			torch.save(model,'./Er_'+str(er)+'Epoch_'+str(epoch)+'.pth')
			np.savetxt('./Er_'+str(er)+'Epoch_'+str(epoch)+'.txt',np.asarray(Er))
			np.savetxt('./Loss_'+str(loss)+'Epoch_'+str(epoch)+'.txt',np.asarray(Loss))
			pass
	torch.save(model,'./Er_'+str(er)+'Epoch_'+str(epoch)+'.pth')
	np.savetxt('./Er_'+str(er)+'Epoch_'+str(epoch)+'.txt',np.asarray(Er))
	np.savetxt('./Loss_'+str(loss)+'Epoch_'+str(epoch)+'.txt',np.asarray(Loss))
	if plotFlag:
		fig=plt.figure()
		ax=plt.subplot(1,1,1)
		ax.plot(Er,label='Relative Error')
		ax.plot(Loss,label='|Residual|')
		ax.legend()
		ax.set_xlabel('Epoch')
		ax.set_yscale('log')
		fig.savefig('./LossResidual.png',bbox_inches='tight')
		plt.show()

	return model,{"Er":np.asarray(Er),"Loss":np.asarray(Loss)}

def trainmodel(DataLoader,LossF,model,optimizer,criterion,qoiidx,softidx,penaltyConstant):
	model.train()
	er_0=0;loss_0=0
	erlist=[]
	ReList=[]
	optimizer.zero_grad()
	for data in DataLoader:
		input=data[0].to('cuda')
		fcn_id=data[0].y[0,0]
		truth=data[0].y[1:,0:]
		fcn=LossF[int(fcn_id)]
		assert (int(fcn_id)-fcn_id)**2<1e-12,\
		'The loss function is selected right!'
		tic=time.time()
		output = model(input)
		Re,dRe,dbc=fcn(output)
		print('wallclock time of evl Res= ',time.time()-tic)
		ReList.append(torch.abs(Re))
		#loss=loss+torch.norm(Re)#criterion(output,truth)#torch.max(torch.abs(Re)) #
		solution=ReshapeFix(torch.clone(output),[len(output.flatten()),1],'C')
		solution[dbc.dbc_idx]=Double(dbc.dbc_val.reshape([len(dbc.dbc_val),1]))
		er_0=er_0+torch.sqrt(criterion(solution,truth)/criterion(truth,truth*0)).item()
		#loss_0=loss_0+loss.item()
		erlist.append(torch.sqrt(criterion(solution,truth)/criterion(truth,truth*0)).item())
	loss=ReList[0]*0
	for i in range(len(ReList)):
		loss=loss+ReList[i]
	print('max Res=',loss.abs().max().item())
	#print(loss)
	loss=torch.norm(loss)#loss.max()#
	if softidx is not None and penaltyConstant is not None:
		print('DataLoss = ',criterion(solution[softidx],truth[softidx])*penaltyConstant)
		loss=criterion(solution[softidx],truth[softidx])*penaltyConstant + loss
		#print(solution-truth)
		#pdb.set_trace()
	else:
		pass
	if qoiidx is not None:
		QOI_ER=torch.sqrt(criterion(solution[qoiidx],truth[qoiidx])\
							/criterion(truth[qoiidx],truth[qoiidx]*0)).item()
		print('QOI Error=',QOI_ER)
		os.system("touch QOIError.txt")
		os.system("touch QOIValue.txt")
		file1 = open("QOIError.txt","a")
		file1.writelines(str(QOI_ER)+"\n")
		file2 = open("QOIValue.txt","a")
		file2.writelines(str(solution[qoiidx].detach().cpu().numpy().reshape([1,-1])[:])+"\n")
		file1.close()
		file2.close()
	else:
		pass
	tic=time.time()
	loss.backward()
	print('wallclock time of this BP= ',time.time()-tic)
	optimizer.step()
	print('>>>>>>>max error<<<<<<< ====================================', max(erlist))
	try:
		print('>>>>>>>model source<<<<<<< =======================',model.source)
		os.system("touch ModelSource.txt")
		file3= open("ModelSource.txt","a")
		object2write=model.source.detach().cpu().numpy().reshape([1,-1])
		for ifer in range(2):
			try:
				file3.writelines(str(object2write[0,ifer])+"\n")
			except:
				pass
		file3.close()
	except:
		pass
	#print('mean error = ',er_0/len(DataLoader))
	#print('mean loss = ',loss.norm().item()/len(DataLoader))
	return er_0/len(DataLoader),loss.norm().item()/len(DataLoader),model



'''
def trainmodel(DataLoader,fcn,model,optimizer,criterion):
	model.train()
	er_0=0;loss_0=0
	batchNum=0
	for data,truth in DataLoader:
		data = data.to('cuda')
		truth = truth.to('cuda').reshape([1,-1,1])
		optimizer.zero_grad()
		output = model(data)
		Re,dRe,dbc=fcn(output)
		loss=torch.norm(Re)#torch.max(torch.abs(Re)) #
		solution=output
		solution[dbc.dbc_idx]=Double(dbc.dbc_val.reshape([len(dbc.dbc_val),1]))
		print('Batch '+str(batchNum)+', Relative Error = ', 
			  torch.sqrt(criterion(solution,truth)/criterion(truth,truth*0)).item())
		print('Batch '+str(batchNum)+', Loss = ',loss.item())
		loss.backward()
		optimizer.step()
		er_0=er_0+torch.sqrt(criterion(solution,truth)/criterion(truth,truth*0)).item()
		loss_0=loss_0+loss.item()
		batchNum=batchNum+1
	return er_0/len(DataLoader),loss_0/len(DataLoader),model
'''	



def Reshape(input, Shape,order='F'):
	if order=='F':
		return torch.reshape(input,[Shape[len(Shape)-1-i] \
			                for i in range(len(Shape))]).permute([len(Shape)-1-i \
							for i in range(len(Shape))])
	elif order=='C':
		return torch.reshape(input, Shape)
	else:
		raise ValueError('Reshape Only Support Fortran or C')


def ReshapeFix(input, Shape,order='F'):
	if order=='F':
		return torch.reshape(input.T,[Shape[len(Shape)-1-i] \
			                for i in range(len(Shape))]).permute([len(Shape)-1-i \
							for i in range(len(Shape))])
	elif order=='C':
		return torch.reshape(input, Shape)
	else:
		raise ValueError('Reshape Only Support Fortran or C')


def Double(A):
	if len(A.shape)==0 or (len(A.shape)==1 and A.shape[0]==1):
		return torch.tensor([A]).reshape([1,1]).double().to('cuda')
	else:		
		return torch.from_numpy(A).double().to('cuda')

def Flatten(A):
	torch.tensor([1,2])