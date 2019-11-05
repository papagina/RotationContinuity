import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import tools

class Model(nn.Module):
    def __init__(self, is_linear=True, out_rotation_mode="ortho6d"):
        super(Model, self).__init__()
        self.out_rotation_mode = out_rotation_mode
        if(self.out_rotation_mode=="ortho6d"):
            self.out_channel = 6
        if(self.out_rotation_mode=="ortho5d"):
            self.out_channel = 5
        if(self.out_rotation_mode=="ortho5d_c3"):
            self.out_channel = 5
        elif (self.out_rotation_mode=="Quaternion"):
            self.out_channel = 4
        elif (self.out_rotation_mode=="Quaternion_half"):
            self.out_channel = 4
        elif (self.out_rotation_mode=="AxisAngle"):
            self.out_channel = 4
        elif (self.out_rotation_mode=="euler"):
            self.out_channel = 3
        elif (self.out_rotation_mode=="euler_sin_cos"):
            self.out_channel = 6
        elif (self.out_rotation_mode=="hopf"):
            self.out_channel = 3
        elif (self.out_rotation_mode=="Rodriguez-vectors"):
            self.out_channel = 3
        
        T_pose_np  = np.array([[1,0,0],[0,1,0], [0,0,1]])
        self.T_pose = torch.autograd.Variable(torch.FloatTensor(T_pose_np).cuda())
        
        self.inner_size =128
        
        if(is_linear==False):
            self.mlp = nn.Sequential(
                    nn.Linear(3*3, self.inner_size),
                    nn.LeakyReLU(),
                    nn.Linear(self.inner_size,self.inner_size),
                    nn.LeakyReLU(),
                    nn.Linear(self.inner_size, self.inner_size),
                    nn.LeakyReLU(),
                    nn.Linear(self.inner_size, self.out_channel)
                    )
        

        else:
            self.mlp = nn.Sequential(
                    nn.Linear(3*3, 128),
                    nn.Linear(128, self.out_channel)
                    )
    
    #batch*3*3
    def set_T_pose(self,T_pose):
        self.T_pose = T_pose

    #in_poses batch*3*3
    def forward(self, in_poses ):
        batch=in_poses.shape[0]
                
        if(self.out_rotation_mode=="ortho6d"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*6
            out_rotation_matrix = tools.compute_rotation_matrix_from_ortho6d(out_rotation_raw) #batch*3*3
        elif(self.out_rotation_mode=="ortho5d"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*5
            out_rotation_matrix = tools.compute_rotation_matrix_from_ortho5d(out_rotation_raw) #batch*3*3
        elif (self.out_rotation_mode=="Quaternion"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*4
            out_rotation_matrix = tools.compute_rotation_matrix_from_quaternion(out_rotation_raw) #batch*3*3
        elif (self.out_rotation_mode=="Quaternion_half"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*4
            #out_rotation_raw_z_abs = torch.abs(out_rotation_raw[:,3]).view(batch,1)
            #out_rotation_raw2= torch.cat( (out_rotation_raw[:,0:3], out_rotation_raw_z_abs),1)#batch*4
            
            sign = torch.sign(out_rotation_raw[:, 3])+0.5 #batch
            out_rotation_raw2  = out_rotation_raw * sign.view(batch,1).repeat(1,4) 
            out_rotation_matrix = tools.compute_rotation_matrix_from_quaternion(out_rotation_raw2) #batch*3*3
        elif (self.out_rotation_mode=="AxisAngle"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*4
            out_rotation_matrix = tools.compute_rotation_matrix_from_axisAngle(out_rotation_raw) #batch*3*3
        elif (self.out_rotation_mode=="euler"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*3
            out_rotation_matrix = tools.compute_rotation_matrix_from_euler(out_rotation_raw) #batch*3*3
        elif (self.out_rotation_mode == "euler_sin_cos"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*3
            out_rotation_raw = torch.tanh(out_rotation_raw)
            out_rotation_matrix = tools.compute_rotation_matrix_from_euler_sin_cos(out_rotation_raw) #batch*3*3
        elif (self.out_rotation_mode=="hopf"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*3
            out_rotation_matrix = tools.compute_rotation_matrix_from_hopf(out_rotation_raw) #batch*3*3
        elif (self.out_rotation_mode=="Rodriguez-vectors"):
            out_rotation_raw =  self.mlp(in_poses.view(batch,-1)) #batch*3
            out_rotation_matrix = tools.compute_rotation_matrix_from_Rodriguez(out_rotation_raw) #batch*3*3
            #print (out_rotation_matrix.shape)

        
        
        out_poses = tools.compute_pose_from_rotation_matrix(self.T_pose, out_rotation_matrix) #batch*joint_num*3
        
        return out_rotation_matrix, out_poses
    
    #gt_poses batch*joint_num*3
    #predict_poses batch*joint_num*3
    def compute_pose_loss(self, gt_poses, out_poses):
        error = torch.pow(gt_poses-out_poses,2).mean() #batch
        
        return error
    
    def compute_geodesic_loss(self, gt_r_matrix, out_r_matrix):
        theta = tools.compute_geodesic_distance_from_two_matrices(gt_r_matrix, out_r_matrix)
        error = theta.mean()
        return error
    
    def compute_quaternion_loss(self, gt_q, out_q):
        error = torch.pow(gt_q - out_q, 2).mean()
        
        return error
