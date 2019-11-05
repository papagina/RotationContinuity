import read_bvh
import os
import numpy as np


def generate_quat_data_from_bvh_data_folder ( src_bvh_folder, tar_quat_folder):
    if not os.path.exists(tar_quat_folder):
        os.makedirs(tar_quat_folder)
    bvh_names=os.listdir(src_bvh_folder)
    total_frames = 0
    for bvh_name in bvh_names:
        if(len(bvh_name)>4):
            if(bvh_name[len(bvh_name)-4: len(bvh_name)]!=".bvh"):
                continue
            quat_fn = tar_quat_folder + bvh_name[0:len(bvh_name)-4] 
            print (quat_fn)
            
            motion = read_bvh.get_quaternion_training_data_from_bvh(src_bvh_folder+ bvh_name)
            print (motion.shape)
            
            np.save(quat_fn , motion)
            total_frames=total_frames+motion.shape[0]
    print ("total frames:")
    print (total_frames)
    return total_frames
    

src_folder = "your folder that contains the bvh files."
tar_folder = "the folder where you output the quaternion format data in npy format."

generate_quat_data_from_bvh_data_folder(src_folder, tar_folder)
