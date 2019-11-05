""" Based on Holden code """
#import tensorflow as tf
import torch
import transforms3d.euler as euler
import numpy as np

import read_bvh_hierarchy
from collections import OrderedDict


    
##FK for quaternions
class FK:
    def __init_(self):
        pass
    
    #output np joint_num*4, the freely rotating joint is 1,1,1,1, the fixed joint is 1,0,0,0
    def get_joint_rotation_matrix_limitaion_mask_np(self, joint_index_lst):
        joint_num=len(joint_index_lst)
        joint_mask = np.ones((joint_num, 4,4))
        for joint in joint_index_lst:
            if("_Nub" in joint):
                index=joint_index_lst[joint]
                joint_mask[index]=np.eye(4,4)
        
        return joint_mask
    
    ##input standard_bvh_fn
    ##ouput (offsets 57*3 numpy, parent_index_lst 42, joint_index_lst ordered_dict end_bones end with "_Nub")
    def get_nonEnd_offsets_and_parent_index_lst(self, standard_bvh_fn):
        skeleton, non_end_bones=read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_fn)
        
        skeleton_non_end = OrderedDict()
        for joint in skeleton.keys():
            if("_Nub" not in joint):
                skeleton_non_end[joint]=skeleton[joint]
        
        skeleton=skeleton_non_end
        
        print (skeleton)

        joint_index_lst = OrderedDict()
        i=0
        for joint in skeleton.keys():
            joint_index_lst[joint]=i
            i=i+1
        
        parent_index_lst=[]
        offsets = []
        for joint in skeleton.keys():
            parent = skeleton[joint]['parent']
            if (parent == None): ##it's hip.  
                parent_index_lst +=[-1]
            else:
                parent_index=joint_index_lst[parent]
                if(parent_index>=len(parent_index_lst)):
                    print ("error! parent's index is equal or smaller than child's index!!")
                parent_index_lst +=[parent_index]
                
            offsets += [skeleton[joint]['offsets']]
        offsets = np.array(offsets)
        return (offsets, parent_index_lst, joint_index_lst)
    
    
    ##input standard_bvh_fn
    ##ouput (offsets 57*3 numpy, parent_index_lst 57, joint_index_lst ordered_dict end_bones end with "_Nub")
    def get_offsets_and_parent_index_lst(self, standard_bvh_fn):
        skeleton, non_end_bones=read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_fn)
        joint_index_lst = OrderedDict()
        i=0
        for joint in skeleton.keys():
            joint_index_lst[joint]=i
            i=i+1
        
        parent_index_lst=[]
        offsets = []
        for joint in skeleton.keys():
            parent = skeleton[joint]['parent']
            if (parent == None): ##it's hip.  
                parent_index_lst +=[-1]
            else:
                parent_index=joint_index_lst[parent]
                if(parent_index>=len(parent_index_lst)):
                    print ("error! parent's index is equal or smaller than child's index!!")
                parent_index_lst +=[parent_index]
                
            offsets += [skeleton[joint]['offsets']]
        offsets = np.array(offsets)
        return (offsets, parent_index_lst, joint_index_lst)
   

    
    #translations batch*joint_num*3
    #matrices batch*joint_num*4*4
    def translations2matrices(self, translations):
        #print (translations)
        batch=translations.shape[0]
        joint_num=translations.shape[1]
        zeros = torch.autograd.Variable(torch.zeros(batch,joint_num,1).cuda())
        ones = torch.autograd.Variable(torch.ones(batch,joint_num,1).cuda())
        
        dim0 = torch.cat( (ones, zeros, zeros, translations[:,:,0].view(batch, joint_num,1))  , 2)
        dim1 = torch.cat( (zeros, ones, zeros, translations[:,:,1].view(batch, joint_num,1))  , 2)
        dim2 = torch.cat( (zeros, zeros, ones, translations[:,:,2].view(batch, joint_num,1))  , 2)
        dim3 = torch.cat( (zeros, zeros, zeros, ones              )  , 2)
        
        matrices=torch.cat((dim0,dim1,dim2, dim3), 2).view(batch, joint_num, 4,4)
        
        return matrices

    #rotation_matrices batch*joint_num*4*4
    #offsets batch*joint_num*3    
    #output batch*joint_num*4*4
    def compute_local_matrices(self, parent_index_lst, rotation_matrices, offsets):
        batch=rotation_matrices.shape[0]
        joint_num=rotation_matrices.shape[1]
        
        translation_matrices=self.translations2matrices(offsets) #batch*joint_num*4*4
           
        matrices = torch.matmul ( translation_matrices.view(-1,4,4 ),  rotation_matrices.view(-1,4,4) )

        return matrices.view(batch,joint_num,4,4)
        
    
    #parent_index_lst: dimension batch*joint_num, the index of the joints' parents. for hip, the parent's index is -1 and its own index is 0
    #                  the parents index values should always be smaller than the children
    #local_offsets: dimension batch*joint_num*3 torch_variable
    #local_rotations: dimenasion batch*joint_num*4*4 torch_variable
    #out global_pos:  batch*joint_num*3
    def compute_global_positions(self, parent_index_lst, local_offsets, local_rotations):
        batch=local_offsets.shape[0]
        joint_num=local_offsets.shape[1]
        
        ##compute local matrix for each joints
        local_matrices = self.compute_local_matrices(parent_index_lst, local_rotations, local_offsets) #batch*joint_num*4*4
        #print (local_matrices[:,:,0:3,3])
        #add hip position to global_joint_positions
        
        global_matrices= local_matrices[:,0,:,:].view(batch,1,4,4) #append hip
        
        #print (global_matrices[:,:,0:3,3])
        
        for i in range(1, joint_num):
            parent_index=parent_index_lst[i]
            
            global_matrix = torch.matmul( global_matrices[:,parent_index] ,  local_matrices[:,i]).view(batch, 1, 4,4) #batch*1*4*4
            
            global_matrices = torch.cat( (global_matrices, global_matrix), 1) #batch*n*4*4
            #print (global_matrices[:,:,0:3,3])
        
        global_pos = global_matrices[:,:,0:3,3]
        
        return global_pos









