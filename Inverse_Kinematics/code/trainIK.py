
import os
import torch
import numpy as np

#import read_bvh
import Model as Model
import trainIK_param as param
from ForwardKinematics import FK
import read_bvh
import tools

import transforms3d.euler as euler

#Seq_len=1024
Joint_num =  57
#In_frame_size = Joints_num*3=171

Skip_initial_frame_num=1



#rotation_matrix_seq seq_len*joint_num*4*4
#hip_pose_seq seq_len*3
def save_network_output_quat_data_to_bvh(hip_pose_seq, rotation_matrix_seq, bvh_fn):
    hip_pose_seq_np = np.array(hip_pose_seq.data.tolist())
    rotation_matrix_seq_np = np.array(rotation_matrix_seq.data.tolist())[:, :, 0:3,0:3]
    read_bvh.write_joint_rotation_matrices_to_bvh(bvh_fn, hip_pose_seq_np, rotation_matrix_seq_np)


#in [(seq_len*(3+joint_num*4))]
#out seq_len*(3+joint_num*4)
#this function will augment both the y-axis rotation and the x,z-axis movements
#the seq_len is determined by the length of the chosen dance file
##the quat_seq from the ground truth is in the world coordinate, while the quat from the predicted hip rotation is in the local coordinate
def get_augmented_gt_seq_for_training(dances_lst, target_framerate, min_seq_len=16, max_seq_len = 128):
    
    dance_id = np.random.randint(0, len(dances_lst))
    dance_quat_seq, frame_rate = dances_lst[dance_id]
    
    dance_quat_seq = dance_quat_seq[Skip_initial_frame_num: dance_quat_seq.shape[0]]

    random_length = np.random.randint(min_seq_len, min(max_seq_len,dance_quat_seq.shape[0]))
    
    random_start_id = np.random.randint(0,dance_quat_seq.shape[0]-random_length)
    
    dance_quat_seq = dance_quat_seq[random_start_id: random_start_id + random_length]
    
    dance_quat_seq = dance_quat_seq[0:int(dance_quat_seq.shape[0]-dance_quat_seq.shape[0]%16)]
    
    stride= int( frame_rate/target_framerate)
   
    dance_quat_seq = dance_quat_seq.reshape(int(dance_quat_seq.shape[0]/stride), stride, dance_quat_seq.shape[1])[:,np.random.randint(0,stride)]
    

    seq_len=dance_quat_seq.shape[0]
    dance_quat_seq_cuda = torch.autograd.Variable(torch.FloatTensor(dance_quat_seq).cuda()) #seq_len*(3+joint_num*4)

    #augment the rotation
    hip_quat_seq_cuda = dance_quat_seq_cuda[:, 3:7]
    hip_matrix_seq_cuda = tools.compute_rotation_matrix_from_quaternion(hip_quat_seq_cuda)#seq_len*3*3
    
    ##generate a random rotation matrix
    axisR= [0,1,0,np.random.randint(0,360)/180*np.pi]
    mat_r=torch.autograd.Variable(torch.FloatTensor(euler.axangle2mat(axisR[0:3], axisR[3])).cuda())#3*3
    mat_r = mat_r.view(1, 3,3).repeat(hip_matrix_seq_cuda.shape[0],1,1) #seq_len*3*3
    
    new_mat_r = torch.matmul(mat_r,hip_matrix_seq_cuda,) #seq_len*3*3
    
    ##get the quaternion seq
    new_hip_quat_seq_cuda = tools.compute_quaternions_from_rotation_matrices(new_mat_r) #seq_len*4
    
    new_hip_pose_seq_cuda = torch.matmul(mat_r, dance_quat_seq_cuda[:,0:3].unsqueeze(2)).view(seq_len,3)
    new_hip_pose_seq_cuda[:,0]=new_hip_pose_seq_cuda[:,0]+np.random.randn(1)[0]
    new_hip_pose_seq_cuda[:,2]=new_hip_pose_seq_cuda[:,2]+np.random.randn(1)[0]
    
    new_dance_quat_seq_cuda = dance_quat_seq_cuda#seq_len*(3+joint_num*4)
    new_dance_quat_seq_cuda[:,0:3]=new_hip_pose_seq_cuda #fix the hip global pose to be at 0
    new_dance_quat_seq_cuda[:,3:7]=new_hip_quat_seq_cuda
    
    
    return new_dance_quat_seq_cuda

def compute_l1_loss_on_parameters(model):
    regularization_loss = 0
    num_param = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
        num_param += param.nelement()
    return regularization_loss/num_param

#input_seq seq_len*in_frame_size
#parent_index_lst 57
#offsets cuda 57*3
def train_one_iteraton(logger, dances_lst,  param, model, optimizer, iteration,  save_bvh_motion=False):

    #print (bone_offsets)
    
    
    optimizer.zero_grad()
    
    
    gt_hip_pose_quat = get_augmented_gt_seq_for_training(dances_lst, param.target_frame_rate,min_seq_len=16, max_seq_len = 64) #seq_len*(3+joint_num*4)
    seq_len = gt_hip_pose_quat.shape[0]
    gt_quat = gt_hip_pose_quat[:, 3:gt_hip_pose_quat.shape[1]].contiguous().view(-1, 4) #(seq_len*joint_num)*4
    gt_rotation_matrices = tools.compute_rotation_matrix_from_quaternion(gt_quat)#seq_len*3*3
    gt_rotation_matrices = tools.get_44_rotation_matrix_from_33_rotation_matrix(gt_rotation_matrices).view(seq_len,-1,4,4)#seq_len*joint_num*4*4
    gt_poses = model.rotation_seq2_pose_seq(gt_rotation_matrices) #seq_len*joint_num*3
    
    #gt_rotation_matrices_fix_hip =torch.cat( (torch.autograd.Variable(torch.eye(4,4).cuda()).view(1,1,4,4).expand(seq_len, 1, 4,4), gt_rotation_matrices[:,1:Joint_num]), 1)
    #gt_poses_fix_hip = model.rotation_seq2_pose_seq(gt_rotation_matrices_fix_hip)
 
    ###network forward########
    #predict_poses, predict_rotation_matrices,  predict_poses_fix_hip, predict_rotation_matrices_fix_hip= model(gt_poses) #seq_len*joint_num*3
    predict_poses, predict_rotation_matrices= model(gt_poses) #seq_len*joint_num*3
    
    ####compute loss##########
    #loss_twist = model.compute_joint_twist_loss(predict_rotation_matrices)
    
    #loss_pose_fix_hip = model.compute_pose_loss(gt_poses_fix_hip, predict_poses_fix_hip)
    #loss_pose_hip = model.compute_pose_loss(gt_poses[:,[1,47,52]], predict_poses[:,[1,47,52]])
    loss_pose = model.compute_pose_loss(gt_poses, predict_poses)
    loss_pose_hip  = model.compute_pose_loss(gt_poses[:,[1,47,52]], predict_poses[:,[1,47,52]])
    if(param.loss_rmat_only_on_hip==1):
        loss_rotation_matrix = model.compute_rotation_matrix_loss(gt_rotation_matrices[:,0], predict_rotation_matrices[:,0])
    else:
        loss_rotation_matrix = model.compute_rotation_matrix_loss(gt_rotation_matrices, predict_rotation_matrices)
    loss = loss_rotation_matrix * param.weight_rotation_matrix + loss_pose * param.weight_pose + loss_pose_hip*param.weight_pose_hip
    #loss_l1_on_parameters = compute_l1_loss_on_parameters(model)
    
    #loss = loss_twist*param.weight_twist + loss_pose*param.weight_pose
    #loss = (loss_pose_hip+loss_pose_fix_hip)*param.weight_pose + loss_l1_on_parameters*0.001 + loss_rotation_matrix *param.weight_rotation_matrix
    
    loss.backward()
    
    optimizer.step()
    
    if(iteration%20==0):    
        logger.add_scalar('loss', loss.item(), iteration)
        #logger.add_scalar('loss_pose_fix_hip',loss_pose_fix_hip.item(),iteration)
        #logger.add_scalar('loss_pose_hip',loss_pose_hip.item(),iteration)
        logger.add_scalar('loss_pose',loss_pose.item(),iteration)
        logger.add_scalar('loss_pose_hip',loss_pose_hip.item(),iteration)
        logger.add_scalar('loss_rotation_matrix', loss_rotation_matrix.item(),iteration)
        #logger.add_scalar('loss_twist',loss_twist.item(),iteration)
    
    if(iteration%100==0):
        print ("############# Iteration "+str(iteration)+" #####################")
        print('loss: ' + str( loss.item()) )
        #print('loss_pose_fix_hip: '+str(loss_pose_fix_hip.item()))
        #print('loss_pose_hip: '+str(loss_pose_hip.item()))
        print('loss_pose: '+str(loss_pose.item()))
        print('loss_pose_hip: '+str(loss_pose_hip.item()))
        print('loss_rotation_matrix: ' + str(loss_rotation_matrix.item()))
        #print('loss_twist: '+str(loss_twist.item()))
            
      
    if((save_bvh_motion==True) ):
        ##save the first motion sequence int the batch.
        
        ##fix the x, z motion of the hip
        hip_pose_seq = gt_hip_pose_quat[:, 0:3]
        hip_pose_seq[:,0]=hip_pose_seq[:,0]*0
        hip_pose_seq[:,2]=hip_pose_seq[:,2]*0 
        
        save_network_output_quat_data_to_bvh(hip_pose_seq, predict_rotation_matrices, param.write_bvh_motion_folder+"%07d"%iteration+"_out.bvh")
       
        save_network_output_quat_data_to_bvh(hip_pose_seq, gt_rotation_matrices, param.write_bvh_motion_folder+"%07d"%iteration+"_gt.bvh")
       


#input dance_folder name
#output a list of dances.
def load_dances(dance_folder, frame_rate=60,target_frame_rate=60, min_dance_seq_len=10):
    dance_files=os.listdir(dance_folder)
    dances=[]
    
    for dance_file in dance_files:
        if( (".npy" in dance_file) == False):
            continue
        dance=np.load(dance_folder+dance_file)
        if(dance.shape[0]>(min_dance_seq_len*frame_rate/target_frame_rate)):
            #print ("load "+dance_file)
            #print ("frame number: "+ str(dance.shape[0]))
            dances=dances+[(dance, frame_rate)]
    return dances

#input [(folder_name, framerate)]
#output [(dance, framerate)]
#dance frame_num * (joint_num*3)
def load_dances_lst(dance_folder_lst, min_dance_seq_len=1100):
    dances_lst =[]
    for (dance_folder, frame_rate) in dance_folder_lst:
        dances=load_dances(dance_folder, frame_rate=frame_rate, target_frame_rate=60,min_dance_seq_len=min_dance_seq_len)
        dances_lst=dances_lst+dances
    return dances_lst



        
        
    
    
# dances_lst: [(dance1, frame_rate), ...]
def train(dances_lst, param):
    torch.cuda.set_device(param.device)
    
    print ("####Initiate model AE")
    
    model = Model.Model(joint_num=57,out_rotation_mode=param.out_rotation_mode)
    if(param.read_weight_path!=""):
        print ("Load "+param.read_weight_path)
        model.load_state_dict(torch.load(param.read_weight_path))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr)#, betas=(0.5,0.9))
    model.train()
    
    model.initialize_skeleton_features("../data/standard.bvh")
   
    print ("start train")
 
    for iteration in range(param.start_iteration,param.total_iteration):   
                
        save_bvh=False
        if(iteration%param.save_bvh_iteration==0):
            save_bvh=True
        train_one_iteraton(param.logger, dances_lst,  param, model, optimizer, 
                           iteration, save_bvh )

        if(iteration%param.save_weight_iteration == 0):
           
            path = param.write_weight_folder + "model_%07d"%iteration 
            #print ("save_weight:  " + path)
            torch.save(model.state_dict(), path+".weight")
            
        if(iteration%10000 == 0):
            path = param.write_weight_folder + "model"
            torch.save(model.state_dict(), path+".weight")


param=param.Parameters()

param.read_config("../training/test0202_ortho6d/test0202_ortho6d.config")

if not os.path.exists(param.write_weight_folder):
    os.makedirs(param.write_weight_folder)
if not os.path.exists(param.write_bvh_motion_folder):
    os.makedirs(param.write_bvh_motion_folder)


dances_lst= load_dances_lst(param.dances_folder_lst, param.min_dance_seq_len)
print ("Loaded "+str(len(dances_lst)) + " dances")

train(dances_lst, param)
#dance_quat_cuda = get_gt_seq_for_training(dances_lst, 60)
#dance_quat = np.array(dance_quat_cuda[0].transpose(0,1).data.tolist())
#MotionAE.read_bvh.write_quaterniondata_to_bvh("test_augment.bvh",dance_quat)




