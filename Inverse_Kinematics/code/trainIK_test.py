
import os
import torch
from torch.autograd import Variable
import numpy as np

#import read_bvh
import Model as Model
import trainIK_param as param
import ForwardKinematics as FK
import read_bvh
import tools
import transforms3d.euler as euler
import matplotlib.pyplot as plt
import ntpath
from matplotlib.ticker import NullFormatter
from scipy import stats

Joint_num =  57

Skip_initial_frame_num=1


#rotation_matrix_seq seq_len*joint_num*4*4
#hip_pose_seq seq_len*3
def save_network_output_quat_data_to_bvh(hip_pose_seq, rotation_matrix_seq, bvh_fn):
    hip_pose_seq_np = np.array(hip_pose_seq.data.tolist())
    rotation_matrix_seq_np = np.array(rotation_matrix_seq.data.tolist())[:, :, 0:3,0:3]

    read_bvh.write_joint_rotation_matrices_to_bvh(bvh_fn, hip_pose_seq_np, rotation_matrix_seq_np)

#output quat seq has augmented global rotation
#out cuda seq_len*(3+joint_num*4)
def get_augmented_dance_seq_from_quat_bvh(dance_quat_seq):
    
    #dance_quat_seq = np.load(quat_npy_fn)
    
    dance_quat_seq = dance_quat_seq[Skip_initial_frame_num: dance_quat_seq.shape[0]] #seq_len*(3+joint_num*4)
    
    seq_len=dance_quat_seq.shape[0]
    dance_quat_seq_cuda = torch.autograd.Variable(torch.FloatTensor(dance_quat_seq).cuda()) #seq_len*(3+joint_num*4)

    #augment the rotation
    hip_quat_seq_cuda = dance_quat_seq_cuda[:, 3:7]
    hip_matrix_seq_cuda = tools.compute_rotation_matrix_from_quaternion(hip_quat_seq_cuda)#seq_len*3*3
    
    ##generate a random rotation matrix
    axisR= [0,1,0,0]#[0,1,0,np.random.randint(0,360)/180.0*np.pi]
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


def test_all_dances(read_weight_path, out_rotation_mode, dance_lst, out_bvh_folder, error_threshold,loss_rmat_only_on_hip =1):
    torch.cuda.set_device(2)
    
    print ("####Initiate model AE")
    
    model = Model.Model(joint_num=57,out_rotation_mode=out_rotation_mode)
    
    print ("Load "+read_weight_path)
    model.load_state_dict(torch.load(read_weight_path))
    model.cuda()
    
    model.initialize_skeleton_features("../data/standard.bvh")
    
    model.eval()
    
    error_lst = np.array([])
    rotation_error_lst =np.array([])
    frame_sum=0
    for dance_raw, framerate,dance_fn in dance_lst:
        for b in range (1):
            dance = get_augmented_dance_seq_from_quat_bvh(dance_raw)
            seq_len = dance.shape[0]
            frame_sum = seq_len+frame_sum
            
            gt_hip_poses = dance[:,0:3] #seq_len*3
            gt_quat = dance[:, 3:dance.shape[1]].contiguous().view(-1,4) #(seq_len*joint_num*3)
            gt_rotation_matrices = tools.compute_rotation_matrix_from_quaternion(gt_quat)#(seq_len*joint_num)*3*3
            gt_rotation_matrices = tools.get_44_rotation_matrix_from_33_rotation_matrix(gt_rotation_matrices).view(seq_len,-1,4,4)#seq_len*joint_num*4*4
            gt_poses = model.rotation_seq2_pose_seq(gt_rotation_matrices) #seq_len*joint_num*3 hip's pose is fixed
            
            ###network forward########
            predict_poses, predict_rotation_matrices  = model(gt_poses) #seq_len*joint_num*3
            
            pose_errors = torch.sqrt(torch.pow(predict_poses-gt_poses,2).sum(2)).mean(1) #seq_len
            error_lst = np.append(error_lst, np.array(pose_errors.data.tolist()))
            
            if(loss_rmat_only_on_hip == 1):
                rotation_errors = tools.compute_geodesic_distance_from_two_matrices(predict_rotation_matrices[:,0],gt_rotation_matrices[:,0])
            else:
                rotation_errors = tools.compute_geodesic_distance_from_two_matrices(predict_rotation_matrices.view(-1,4,4),gt_rotation_matrices.view(-1,4,4)).view(seq_len, Joint_num).mean(1)
            rotation_errors = rotation_errors*180/np.pi
            rotation_error_lst= np.append(rotation_error_lst, np.array(rotation_errors.data.tolist()))
            
            ###write the predicted motion with big error
            #if(rotation_errors.max()>error_threshold):
            if(pose_errors.max()>error_threshold):    
                #worst_frame_id = rotation_errors.argmax().item()
                worst_frame_id = pose_errors.argmax().item()
                start_id = max(0,worst_frame_id - 20)
                end_id = min(gt_hip_poses.shape[0], worst_frame_id+20 )

                gt_hip_poses[:, 0] = gt_hip_poses[:,0]*0
                gt_hip_poses[:, 2] = gt_hip_poses[:,2]*0
                head, tail = ntpath.split(dance_fn)
                head2,tail2 = ntpath.split(dance_fn[0:len(dance_fn)-len(tail)-1])
                out_bvh_folder2 = out_bvh_folder+tail2+"/"
                if not os.path.exists(out_bvh_folder2):
                    os.makedirs(out_bvh_folder2)
                out_fn=out_bvh_folder2+tail[0:len(tail)-4]
                print (tail2+"/"+tail[0:len(tail)-4])
                print ("worst frame id: "+str(worst_frame_id))
                print ("error: "+str(pose_errors[start_id:end_id].max().item()))
                #print ("worst frame id: "+str(rotation_errors[start_id:end_id].argmax().item()))
                #print ("error: "+str(rotation_errors[start_id:end_id].max().item()))
                
                

                save_network_output_quat_data_to_bvh(gt_hip_poses[start_id:end_id], predict_rotation_matrices[start_id:end_id], out_fn + "_out.bvh")
                save_network_output_quat_data_to_bvh(gt_hip_poses[start_id:end_id], gt_rotation_matrices[start_id:end_id], out_fn + "_gt.bvh")
        
    avg_error = error_lst.mean()
    max_error = error_lst.max()
    print ("avg error: " + str(avg_error))
    print ("max error: " + str(max_error))   
    avg_rotation_error = rotation_error_lst.mean()
    max_rotation_error = rotation_error_lst.max()
    print ("avg rotation error: " + str(avg_rotation_error))
    print ("max rotation error: " + str(max_rotation_error))  
    return error_lst, rotation_error_lst
    
        


#errror_list np 
def visualize_error_lst(error_lst, histogram_bins_num):
    plt.hist(error_lst,  bins=histogram_bins_num, log=True)
    plt.ylabel('Pose_errors')
    plt.legend(loc='upper right')
    plt.show()    
        
    plt.clf()


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
            dances=dances+[(dance, frame_rate, dance_folder+dance_file)]
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



def get_error_lst(dance_lst, weight_folder, model_name_lst,  iteration, error_threshold, sample_num):
    errors_lst=[]
    for (weight_sub_folder, out_rotation_mode) in model_name_lst:
        weight_fn  = weight_folder + weight_sub_folder+"/weight/model_%07d.weight"%iteration
        out_bvh_folder = weight_folder + weight_sub_folder+"/test_"+"%07d"%iteration+"/"
        if not os.path.exists(out_bvh_folder):
            os.makedirs(out_bvh_folder)
        with torch.no_grad():
            #pose_errorss=[]
            for n in range(sample_num):
                pose_errors, r_errors = test_all_dances(weight_fn, out_rotation_mode, dance_lst, out_bvh_folder, error_threshold) 
                #pose_errorss = pose_errorss + [pose_errors]
        errors_lst = errors_lst + [(out_rotation_mode, pose_errors)]
    return errors_lst


dances_folder_lst = [("../data/quat_data_test/", 60)]

dance_lst= load_dances_lst(dances_folder_lst,min_dance_seq_len=10)
print ("Loaded "+str(len(dance_lst)) + " dances")





weight_folder= "../training/"
model_name_lst = [ ("test0202_ortho6d","ortho6d"),
                   ("test0204_ortho5d","ortho5d"),
                   ("test0201_quat","Quaternion"),
                   ("test0205_axisAngle","AxisAngle"),
                   ("test0203_euler","euler"),
                   ("test0206_mat","rmat")]

iteration=1960000

errors_lst = get_error_lst(dance_lst, weight_folder,model_name_lst,  iteration, error_threshold=0.5, sample_num=3)
print (errors_lst)











