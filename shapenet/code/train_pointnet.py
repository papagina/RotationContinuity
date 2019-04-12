import torch
import torch.nn as nn
import numpy as np
import tools
import Model_pointnet
import pts_loader
import train_pointnet_params as Params

import os




def train_one_iteraton( pc_fn_lst,  param, model, optimizer, iteration):
    
    optimizer.zero_grad()
    batch=param.batch
 
    ###get training data######
    pc_id = np.random.randint(0, len(pc_fn_lst))
    pc1_np  = np.array(pts_loader.load(pc_fn_lst[pc_id]))
    point_num = int(pc1_np.shape[0]/2)
    pc1 = torch.autograd.Variable(torch.FloatTensor(pc1_np[0:point_num]).cuda()) #num*3
    
    pc1 = pc1.view(1, point_num,3).expand(batch,point_num,3).contiguous() #batch*p_num*3
    gt_rmat = tools.get_sampled_rotation_matrices_by_axisAngle(batch)#batch*3*3
    
    gt_rmats = gt_rmat.contiguous().view(batch,1,3,3).expand(batch, point_num, 3,3 ).contiguous().view(-1,3,3)
    pc2 = torch.bmm(gt_rmats, pc1.view(-1,3,1))#(batch*point_num)*3*1
    pc2 = pc2.view(batch, point_num, 3) ##batch*p_num*3
    
    if(model.regress_t==True):
        gt_translation = torch.autograd.Variable(torch.FloatTensor(np.random.randn(batch,3)).cuda())/10.0
        pc2 = pc2 + gt_translation.view(batch,1,3).expand(batch, point_num, 3)
    
    ###network forward########
    if(model.regress_t == True):
        out_rmat, out_translation, out_pc2 = model(pc1, pc2)#batch*3*3
    else:
        out_rmat, out_pc2 = model(pc1, pc2)#batch*3*3
    ####compute loss##########
    
    loss_rmat = model.loss_rmat(gt_rmat, out_rmat)
    loss = loss_rmat * param.weight_rmat
    #loss_geodesic = model.loss_geodesic(gt_rmat, out_rmat)
    
    #loss_pose = model.loss_pose(pc2, out_pc2)
    
    if(model.regress_t==True):
        loss_translation = model.loss_t(gt_translation, out_translation)
        loss = loss + loss_translation*param.weight_translation
    
    loss.backward()
    
    optimizer.step()
    
    if(iteration%10==0):    
        param.logger.add_scalar('loss', loss.item(), iteration)
        #param.logger.add_scalar('loss_pose',loss_pose.item(),iteration)
        param.logger.add_scalar('loss_rmat', loss_rmat.item(),iteration)
        #param.logger.add_scalar('loss_geodesic', loss_geodesic.item(),iteration)
        if(model.regress_t==True):
            param.logger.add_scalar('loss_translation', loss_translation.item(),iteration)
    
    if(iteration%100==0):
        print ("############# Iteration "+str(iteration)+" #####################")
        print('loss: ' + str( loss.item()) )
        #print('loss_pose: '+str(loss_pose.item()))
        print('loss_rmat: ' + str(loss_rmat.item()))
        #print('loss_geodesic: ' + str(loss_geodesic.item()))
        if(model.regress_t==True):
            print('loss_translation: '+str(loss_translation.item()))



#input [folder_name]
#output [point_cloud]
#point_cloud num*3
def load_pc_fn_lst(pc_folder_lst):
    pc_fn_lst =[]
    for pc_folder in pc_folder_lst:
        pc_names = os.listdir(pc_folder)
        for pc_name in pc_names:
            pc_fn_lst = pc_fn_lst +[pc_folder+"/"+pc_name] 
            
    return pc_fn_lst
        
        
# pc_lst: [point_num*3]
def train(pc_lst, param):
    torch.cuda.set_device(param.device)
    
    print ("####Initiate model AE")
    
    model = Model_pointnet.Model(out_rotation_mode=param.out_rotation_mode, regress_t=param.regress_t)
    if(param.read_weight_path!=""):
        print ("Load "+param.read_weight_path)
        model.load_state_dict(torch.load(param.read_weight_path))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr)#, betas=(0.5,0.9))
    model.train()
    
    
    print ("start train")
 
    for iteration in range(param.start_iteration,param.total_iteration):   
                
        train_one_iteraton(pc_lst,  param, model, optimizer, iteration)

        if(iteration%param.save_weight_iteration == 0):
           
            path = param.write_weight_folder + "model_%07d"%iteration 
            #print ("save_weight:  " + path)
            torch.save(model.state_dict(), path+".weight")
            
        if(iteration%10000 == 0):
            path = param.write_weight_folder + "model"
            torch.save(model.state_dict(), path+".weight")


param=Params.Parameters()

#param.read_config("../train/test0201_plane_quat/test0201_plane_quat.config")
#param.read_config("../train/test0202_plane_ortho6d/test0202_plane_ortho6d.config")
#param.read_config("../train/test0203_plane_rmat/test0203_plane_rmat.config")
#param.read_config("../train/test0204_plane_axisAngle/test0204_plane_axisAngle.config")
#param.read_config("../train/test0205_plane_euler/test0205_plane_euler.config")
#param.read_config("../train/test0301_plane_ortho6d/test0301_plane_ortho6d.config")
#param.read_config("../train/test0302_plane_rmat/test0302_plane_rmat.config")
#param.read_config("../train/test0303_plane_quat/test0303_plane_quat.config")
#param.read_config("../train/test0304_plane_axisAngle/test0304_plane_axisAngle.config")
#param.read_config("../train/test0305_plane_euler/test0305_plane_euler.config")
param.read_config("../train/test0306_plane_ortho5d/test0306_plane_ortho5d.config")
if not os.path.exists(param.write_weight_folder):
    os.makedirs(param.write_weight_folder)

pc_fn_lst= load_pc_fn_lst(param.shape_folder_lst) #[3*num]

print ("Train set: "+str(len(pc_fn_lst)) + " shapes")

train(pc_fn_lst, param)



