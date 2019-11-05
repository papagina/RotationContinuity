import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import tools
import matplotlib.pyplot as plt
from model import Model



        
def test(model,  sample_num, input_mode, histogram_bins_num, angle_group_num):
    print ("########TEST##########")
    with torch.no_grad():
        model.eval()
        gt_rotation_matrix = tools.get_sampled_rotation_matrices_by_axisAngle(sample_num) #batch*3*3
        gt_poses = tools.compute_pose_from_rotation_matrix(model.T_pose, gt_rotation_matrix) #batch*3*3
    
        if(input_mode == "pose"):
            out_rotation_matrix, out_poses  = model(gt_poses)
        elif(input_mode == "r_matrix"):
            out_rotation_matrix, out_poses  = model(gt_rotation_matrix)
            
        #print (out_rotation_matrix[0])
        #print (out_poses[0])
        pose_errors = torch.pow(gt_poses-out_poses, 2).sum(2).mean(1) #batch
        pose_error_order = np.argsort(np.array(pose_errors.data.tolist()))
        geodesic_errors= tools.compute_geodesic_distance_from_two_matrices(out_rotation_matrix, gt_rotation_matrix)
        geodesic_error_order = np.argsort(np.array(geodesic_errors.data.tolist()))
    
        
        print ("avg pose error: "+str(pose_errors.mean().item()))
        print ("max pose error: "+str(pose_errors.max().item()))
        print ("avg geodesic error: "+str(geodesic_errors.mean().item()))
        print ("max geodesic error: " + str(geodesic_errors.max().item()))
        
        
        
        
        ####compute the rotation angles
        thetas = tools.compute_angle_from_r_matrices(gt_rotation_matrix)
        
        ####print angles with worst errors######################
        str_angles="("+str(pose_errors[pose_error_order[sample_num-20]].item())+", "+str(pose_errors[pose_error_order[sample_num-1]].item())+")\n"
        for i in range(sample_num-20, sample_num):
            str_angles=str_angles+ str(int(thetas[pose_error_order[i]].item()/np.pi*180))+" "
        print ("angles of max pose errors: "+str_angles)
        str_angles="("+str(geodesic_errors[geodesic_error_order[sample_num-20]].item())+" "+str(geodesic_errors[geodesic_error_order[sample_num-1]].item())+")\n"
        for i in range(sample_num-20, sample_num):
            str_angles=str_angles+ str(int(thetas[geodesic_error_order[i]].item()/np.pi*180))+" "
        print ("angles of max pose errors: "+str_angles)
        
        
        ###visualize the error histogram######################
        ####group errors by theta
        
        group_num=angle_group_num
        
        #angle_bins =[ [0,np.pi/32],[15.5*np.pi/32, 16.5*np.pi/32], [31*np.pi/32, np.pi]] 
        #group_num=3
        angle_bins=[[0,np.pi]]
        group_num=1
        pose_error_lst = []
        geodesic_error_lst = []
        for [min_angle, max_angle] in angle_bins:
            pose_error= []
            geodesic_error =[]
            for j in range(sample_num):
                if (thetas[j]<max_angle) and (thetas[j]>=min_angle):
                    pose_error = pose_error+[pose_errors[j].item()]
                    geodesic_error = geodesic_error + [geodesic_errors[j].item()]
            
            pose_error_lst = pose_error_lst + [np.array(pose_error).astype(float)]
            geodesic_error_lst = geodesic_error_lst+[np.array(geodesic_error).astype(float)]
        
        
        
        for i in range(group_num):
            pose_error= pose_error_lst[i]
            print ("group"+str(i)+" sample num:"+str(pose_error.shape[0]))
            min_angle = int(angle_bins[i][0]/np.pi*180)
            max_angle =  int(angle_bins[i][1]/np.pi*180)
            
            plt.hist([pose_error],  bins=histogram_bins_num, log=True, alpha=0.5, label=str(min_angle)+"_"+str(max_angle))
        plt.ylabel('Pose_errors')
        plt.legend(loc='upper right')
        plt.show()    
        
        plt.clf()
        
        for i in range(group_num):
            geodesic_error= geodesic_error_lst[i]
            print ("group"+str(i)+" sample num:"+str(geodesic_error.shape[0]))
            min_angle = int(angle_bins[i][0]/np.pi*180)
            max_angle =  int(angle_bins[i][1]/np.pi*180)
            
            plt.hist([geodesic_error], bins=histogram_bins_num, log=True, alpha=0.5, label=str(min_angle)+"-"+str(max_angle))
        plt.ylabel('Geodesic_errors')
        plt.legend(loc='upper right')
        plt.show() 
        
        plt.clf()
        
    model.train()
    

    

def train(model, input_mode = "pose", loss_mode="pose", sampling_method = "axis_angle", batch=64, total_iter=20000, out_weight_folder="../train/"):
    if not os.path.exists(out_weight_folder):
        os.makedirs(out_weight_folder)
    
    batch = 64
    
    print ("####Initiate model AE")
    
    #model.double()
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)#, betas=(0.5,0.9))
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    model.train()
    
    T_pose_np  = np.array([[1,0,0],[0,1,0], [0,0,1]])
    T_pose = torch.autograd.Variable(torch.FloatTensor(T_pose_np).cuda())
    model.set_T_pose(T_pose)
    for iteration in range(total_iter):
        
        if(iteration == 10000):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.000001
        
        ##prepare training data
        gt_rotation_matrix = []
        if(sampling_method=="axis_angle"):  
            gt_rotation_matrix = tools.get_sampled_rotation_matrices_by_axisAngle(batch) #batch*3*3
        elif(sampling_method=="quaternion"):
            gt_rotation_matrix = tools.get_sampled_rotation_matrices_by_quat(batch) #batch*3*3
            
        gt_poses = tools.compute_pose_from_rotation_matrix(T_pose, gt_rotation_matrix) #batch*3*3
        
        optimizer.zero_grad()
        
        if(input_mode == "pose"):
            out_rotation_matrix, out_poses  = model(gt_poses)
        elif(input_mode == "r_matrix"):
            out_rotation_matrix, out_poses  = model(gt_rotation_matrix)
            
        if(loss_mode == "pose"): 
            loss = model.compute_pose_loss(gt_poses, out_poses)
        elif(loss_mode == "geodesic"):
            loss = model.compute_geodesic_loss(gt_rotation_matrix, out_rotation_matrix)
                
        loss.backward()
    
        optimizer.step()
    
        if(iteration%5000 == 0):
            print ("######## iteration " +str(iteration))
            print ("loss:" + str(loss.item()))
        
        if(iteration%10000 == 0):
            path = out_weight_folder + "_%07d"%iteration+".weight"
            torch.save(model.state_dict(), path)
        
        #if(iteration%5000 == 0):
    test(model, sample_num=2048, input_mode=input_mode, histogram_bins_num=10,angle_group_num=30)


out_weight_folder = "../train/test0000/"


torch.cuda.set_device(0)

##the "loss_mode" of "pose" just means computing the l2 loss between the final output rotation matrix and the groundtruth matrix given line 125

print ("################TEST ON ortho6d, input=r_matrix, loss=pose#####################")
model_o6mp = Model(is_linear=False, out_rotation_mode="ortho6d")
train(model_o6mp, input_mode = "r_matrix", loss_mode="pose", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"o6mp/")
print ("################TEST ON ortho6d, input=r_matrix, loss=Geodesic#####################")
model_o6mg = Model(is_linear=False, out_rotation_mode="ortho6d")
train(model_o6mg, input_mode = "r_matrix", loss_mode="geodesic", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"o6mg/")


print ("################TEST ON ortho5d, input=r_matrix, loss=pose#####################")
model_o5pp = Model(is_linear=False, out_rotation_mode="ortho5d")
train(model_o5pp, input_mode = "r_matrix", loss_mode="pose", sampling_method="quaternion", batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"o5mp/")
print ("################TEST ON ortho5d, input=pose, loss=Geodesic#####################")
model_o5pg = Model(is_linear=False, out_rotation_mode="ortho5d")
train(model_o5pg, input_mode = "r_matrix", loss_mode="geodesic", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"o5mg/")


print ("################TEST ON Quaternion, input=r_matrix, loss=pose#####################")
model_qmp = Model(is_linear=False, out_rotation_mode="Quaternion")
train(model_qmp, input_mode = "r_matrix", loss_mode="pose", batch=64 , sampling_method="quaternion", total_iter=500001, out_weight_folder=out_weight_folder+"qmp/")
print ("################TEST ON Quaternion, input=r_matrix, loss=geodesic#####################")
model_qmg = Model(is_linear=False, out_rotation_mode="Quaternion")
train(model_qmg, input_mode = "r_matrix", loss_mode="geodesic", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"qmg/")

print ("################TEST ON AxisAngle, input=r_matrix, loss=pose#####################")
model_amp = Model(is_linear=False, out_rotation_mode="AxisAngle")
train(model_amp, input_mode = "r_matrix", loss_mode="pose", batch=64 , sampling_method="quaternion", total_iter=500001, out_weight_folder=out_weight_folder+"amp/")
print ("################TEST ON AxisAngle, input=r_matrix, loss=geodesic#####################")
model_amg = Model(is_linear=False, out_rotation_mode="AxisAngle")
train(model_amg, input_mode = "r_matrix", loss_mode="geodesic", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"amg/")


print ("################TEST ON euler, input=r_matrix, loss=pose#####################")
model_emp = Model(is_linear=False, out_rotation_mode="euler")
train(model_emp, input_mode = "r_matrix", loss_mode="pose", batch=64 , sampling_method="quaternion", total_iter=500001, out_weight_folder=out_weight_folder+"emp/")
print ("################TEST ON euler, input=r_matrix, loss=geodesic#####################")
model_emg = Model(is_linear=False, out_rotation_mode="euler")
train(model_emg, input_mode = "r_matrix", loss_mode="geodesic", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"emg/")

print ("################TEST ON Rodriguez-vectors, input=r_matrix, loss=pose#####################")
model_rmp = Model(is_linear=False, out_rotation_mode="Rodriguez-vectors")
train(model_emp, input_mode = "r_matrix", loss_mode="pose", batch=64 , sampling_method="quaternion", total_iter=500001, out_weight_folder=out_weight_folder+"rmp/")
print ("################TEST ON Rodriguez-vectors, input=r_matrix, loss=geodesic#####################")
model_rmg = Model(is_linear=False, out_rotation_mode="Rodriguez-vectors")
train(model_emg, input_mode = "r_matrix", loss_mode="geodesic", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"rmg/")

print ("################TEST ON euler_sin_cos, input=r_matrix, loss=pose#####################")
model_escmp = Model(is_linear=False, out_rotation_mode="euler_sin_cos")
train(model_emp, input_mode = "r_matrix", loss_mode="pose", batch=64 , sampling_method="quaternion", total_iter=500001, out_weight_folder=out_weight_folder+"escmp/")
print ("################TEST ON euler_sin_cos, input=r_matrix, loss=geodesic#####################")
model_escmg = Model(is_linear=False, out_rotation_mode="euler_sin_cos")
train(model_emg, input_mode = "r_matrix", loss_mode="geodesic", sampling_method="axis_angle",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"escmg/")

print ("################TEST ON Quaternion_half, input=r_matrix, loss=geodesic#####################")
model_qhmp = Model(is_linear=False, out_rotation_mode="Quaternion_half")
train(model_emg, input_mode = "r_matrix", loss_mode="pose", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"qhmp/")
model_qhmg = Model(is_linear=False, out_rotation_mode="Quaternion_half")
train(model_emg, input_mode = "r_matrix", loss_mode="geodesic", sampling_method="quaternion",batch=64 , total_iter=500001, out_weight_folder=out_weight_folder+"qhmg/")


    
    
