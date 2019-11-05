import torch
import torch.nn as nn
import numpy as np
import read_bvh
from ForwardKinematics import FK
import tools



#Joints_num =  57
#In_frame_size = Joints_num*3= 171
class Model(nn.Module):
    def __init__(self, joint_num=57,out_rotation_mode="Quaternion"):
        super(Model, self).__init__()
        
        self.joint_num=joint_num
        
        self.out_rotation_mode = out_rotation_mode
        
        self.joint_parent_matrix=torch.autograd.Variable(torch.Tensor(read_bvh.parenting_matrix).cuda())
        
        if(self.out_rotation_mode=="ortho6d"):
            self.out_channel = 6
        if(self.out_rotation_mode=="ortho5d"):
            self.out_channel = 5
        elif (self.out_rotation_mode=="Quaternion"):
            self.out_channel = 4
        elif (self.out_rotation_mode=="AxisAngle"):
            self.out_channel = 4
        elif (self.out_rotation_mode=="euler"):
            self.out_channel = 3
        elif (self.out_rotation_mode=="hopf"):
            self.out_channel = 3
        elif (out_rotation_mode  == "rmat"):
            self.out_channel = 9
            
        #if(self.hip_5d_rotation==True):
        #    self.out_channel = 5+(joint_num-1)*3
        self.mlp = nn.Sequential(
                nn.Linear(self.joint_num*3, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, self.out_channel*joint_num)
                )
    
    #lst joint_num
    def set_parent_index_lst(self,parent_index_lst):
        self.parent_index_lst = parent_index_lst
    
    #numpy joint_num*3
    def set_joint_offsets(self, joint_offsets):
        self.joint_offsets = joint_offsets
    
    
    def initialize_skeleton_features(self, standard_bvh_fn):
        fk=FK()
        joint_offsets, parent_index_lst, joint_index_lst = fk.get_offsets_and_parent_index_lst(standard_bvh_fn)
        joint_offsets=joint_offsets*0.01
        self.set_parent_index_lst(parent_index_lst)
        self.set_joint_offsets(joint_offsets)
        
        #self.joint_rotation_matrix_limitation_mask = torch.autograd.Variable(torch.FloatTensor(fk.get_joint_rotation_matrix_limitaion_mask_np(joint_index_lst)).cuda()) #joint_num*4*4
        #self.joint_rotation_matrix_limitation_mask = self.joint_rotation_matrix_limitation_mask.view(1,self.joint_num, 4,4)
    
    
    
    
    #joint_rotation_matrices batch*joint_num*4*4  
    #the output pose will be fixed at the hip position
    #pose_batch batch*joint_num*3
    def rotation_seq2_pose_seq(self, joint_rotation_matrices):
        joint_num=self.joint_num
        batch=joint_rotation_matrices.shape[0]
        
        
        ##get the offsets batch
        offsets_batch = torch.autograd.Variable(torch.FloatTensor(self.joint_offsets).cuda())
        offsets_batch = offsets_batch.view(1, joint_num,3).repeat(batch, 1,1) #batch*joint_num*3
        
        fk=FK()
        
        
        
        pose_batch=fk.compute_global_positions(self.parent_index_lst, offsets_batch, joint_rotation_matrices)#batch*joint_num*3
        return pose_batch.contiguous()
        
    
    #in_seq  b*joint_num*3
    #out_poses b*joint_num*3
    #out_rotation_matrix b*joint_num*4*4
    def forward(self, in_seq):
        batch=in_seq.shape[0]
        
        if(self.out_rotation_mode == "Quaternion"):
            out_r = self.mlp(in_seq.view(batch,-1)) #batch*(joint_num*4)
            out_rotation_matrices = tools.compute_rotation_matrix_from_quaternion(out_r.view(-1, self.out_channel))#(batch*joint_num)*3*3
        elif (self.out_rotation_mode == "ortho6d"):
            out_r = self.mlp(in_seq.view(batch,-1)) #batch*(joint_num*6)
            out_rotation_matrices = tools.compute_rotation_matrix_from_ortho6d(out_r.view(-1, self.out_channel))
        elif (self.out_rotation_mode == "ortho5d"):
            out_r = self.mlp(in_seq.view(batch,-1)) #batch*(joint_num*5)
            out_rotation_matrices = tools.compute_rotation_matrix_from_ortho5d(out_r.view(-1, self.out_channel))
        elif (self.out_rotation_mode == "AxisAngle"):
            out_r = self.mlp(in_seq.view(batch,-1)) #batch*(joint_num*4)
            out_rotation_matrices = tools.compute_rotation_matrix_from_axisAngle(out_r.view(-1, self.out_channel))
        elif (self.out_rotation_mode == "euler"):
            out_r = self.mlp(in_seq.view(batch,-1)) #batch*(joint_num*3)
            out_rotation_matrices = tools.compute_rotation_matrix_from_euler(out_r.view(-1, self.out_channel))
        elif ((self.out_rotation_mode=="rmat") and (self.training==True)):
            out_r = self.mlp(in_seq.view(batch,-1)) #batch*(joint_num*9)
            out_rotation_matrices = out_r.view(-1,3,3)#(b*joint_num)*3*3
        elif ((self.out_rotation_mode=="rmat") and (self.training==False)):
            out_r = self.mlp(in_seq.view(batch,-1)) #batch*(joint_num*9)
            out_rotation_matrices = out_r.view(-1,3,3)#(b*joint_num)*3*3
            out_rotation_matrices = tools.compute_rotation_matrix_from_matrix(out_rotation_matrices)

        out_rotation_matrices = tools.get_44_rotation_matrix_from_33_rotation_matrix(out_rotation_matrices).view(batch,self.joint_num,4,4)#(batch*joint_num)*4*4       
        #out_rotation_matrices = out_rotation_matrices * self.joint_rotation_matrix_limitation_mask.expand(batch, self.joint_num,4,4)
        out_poses = self.rotation_seq2_pose_seq(out_rotation_matrices) #b*joint_num*3
        
        #out_rotation_matrices_fix_hip =torch.cat( (torch.autograd.Variable(torch.eye(4,4).cuda()).view(1,1,4,4).expand(batch, 1, 4,4), out_rotation_matrices[:,1:self.joint_num]), 1)
        #out_poses_fix_hip = self.rotation_seq2_pose_seq(out_rotation_matrices_fix_hip)
        
        #out_rotation_matrices_with_hip = torch.cat((out_rotation_matrices[:,0:1], out_rotation_matrices[:,1:self.joint_num].detach()), 1)
        #out_poses_with_hip = self.rotation_seq2_pose_seq(out_rotation_matrices_with_hip)
        
        return out_poses, out_rotation_matrices#out_poses_with_hip, out_rotation_matrices_with_hip , out_poses_fix_hip, out_rotation_matrices_fix_hip


    
    #in cuda tensor  b*joint_num*3 b*joint_num*3
    def compute_pose_loss(self, gt_pose_seq, predict_pose_seq):
        loss_function= torch.nn.MSELoss()#torch.nn.MSELoss()
        loss = loss_function(predict_pose_seq, gt_pose_seq)
        return loss
    
    #in cuda b*joint_num*4*4
    def compute_rotation_matrix_loss(self, gt_rotation_matrix, predict_rotation_matrix):
        loss_function = torch.nn.MSELoss()
        loss = loss_function(predict_rotation_matrix, gt_rotation_matrix)
        return loss
 
    
    #predict_rotation_matrices batch*joint_num*4*4 these are local euler rotation defined on the coordinate on joints.
    def compute_joint_twist_loss(self, predict_rotation_matrices):
        batch=predict_rotation_matrices.shape[0]
        joint_num=predict_rotation_matrices.shape[1]
        
        #eliminate the hip's rotation
        r_matrices = predict_rotation_matrices[:,1:joint_num].contiguous().view(-1,4,4) #(batch*(joint_num-1))*4*4
        eulers =tools.compute_euler_angles_from_rotation_matrices(r_matrices) #(batch*(joint_num-1))*3
        
        zeros =torch.autograd.Variable(torch.zeros(batch*(joint_num-1)).cuda())
        threshold = zeros + 100/180*np.pi
        
        loss_twist = torch.pow(torch.max(zeros, torch.abs(eulers[:,1])-threshold),2).mean() 
        
        return loss_twist
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
