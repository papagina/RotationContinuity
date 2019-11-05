import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import tools 
from model import Model


#numpy sample_num
def test(model,  sample_num, input_mode, sample_method="axisAngle"):
    #print ("########TEST##########")
    with torch.no_grad():
        model.eval()
        model.cuda()
        gt_rotation_matrix =[]
        if(sample_method == "axisAngle"):   
            gt_rotation_matrix = tools.get_sampled_rotation_matrices_by_axisAngle(sample_num) #batch*3*3
        elif(sample_method == "quaternion"):   
            gt_rotation_matrix = tools.get_sampled_rotation_matrices_by_quat(sample_num) #batch*3*3
        gt_poses = tools.compute_pose_from_rotation_matrix(model.T_pose, gt_rotation_matrix) #batch*3*3
        
        if(input_mode == "pose"):
            out_rotation_matrix, out_poses  = model(gt_poses)
        elif(input_mode == "r_matrix"):
            out_rotation_matrix, out_poses  = model(gt_rotation_matrix)

        #print (out_rotation_matrix[0])
        #print (out_poses[0])
        #pose_errors = torch.pow(gt_poses-out_poses, 2).sum(2).mean(1) #batch
        #pose_error_order = np.argsort(np.array(pose_errors.data.tolist()))
        geodesic_errors= tools.compute_geodesic_distance_from_two_matrices(out_rotation_matrix, gt_rotation_matrix)
        geodesic_errors = geodesic_errors/np.pi*180
        geodesic_errors = np.array(geodesic_errors.data.tolist())
        
        
        #geodesic_error_order = np.argsort(np.array(geodesic_errors.data.tolist()))
    return geodesic_errors  

    
    
def get_error_lst(model_name_lst, weight_folder, iteration, sample_num,sample_method = "axisAngle"):
    errors_lst=[]
    for model_name in model_name_lst:
        if(model_name[0]=="a"):
            out_rotation_mode = "AxisAngle"
        if(model_name[0]=="r"):
            out_rotation_mode = "Rodriguez-vectors"
        elif(model_name[0]=="e"):
            out_rotation_mode="euler"
            if(model_name[0:3]=="esc"):
                out_rotation_mode = "euler_sin_cos"
        elif(model_name[0]=="q"):
            out_rotation_mode="Quaternion"
            if(model_name[0:2]=="qh"):
                out_rotation_mode = "Quaternion_half"
        elif(model_name[0]=="o"):
            if(model_name[1]=="5"):
                out_rotation_mode="ortho5d"
            elif(model_name[1]=="6"):
                out_rotation_mode ="ortho6d"
        #print (out_rotation_mode)
        model = Model(is_linear = False, out_rotation_mode = out_rotation_mode)
        weight_fn = weight_folder + model_name +"/_%07d"%iteration +".weight"
        model.load_state_dict(torch.load(weight_fn))
        if(model_name[len(model_name)-2]=="m"):
            input_mode = "r_matrix"
        elif(model_name[len(model_name)-2]=="p"):
            input_mode = "pose"
        errors = test(model, sample_num=sample_num, input_mode=input_mode, sample_method=sample_method)
        errors_lst = errors_lst+[(errors, model_name)]
    return errors_lst


torch.cuda.set_device(1)

weight_folder= "../train/test0000/"

#model with geodesic loss
model_name_lst_g = [ "o6mg", "o5mg", "qmg", "qhmg", "amg","rmg", "emg", "escmg"]
#models with l2 loss on rotation matrices
model_name_lst_p = [ "o6mp", "o5mp", "qmp", "qhmp", "amp","rmp", "emp", "escmp"]

model_name_lst = model_name_lst_g+model_name_lst_p

iteration=500000

sample_method = "quaternion"   
errors_lst = get_error_lst(model_name_lst, weight_folder, iteration, sample_num=100000,sample_method=sample_method)

for errors, name in errors_lst:
    print (name)
    print ("mean:" + str( np.round( errors.mean(),2) ))
    print ("max:" + str( np.round(errors.max(),2) ))
    print ("std:"+  str( np.round(errors.std(),2) ))





