import tensorboardX
from datetime import date
import configparser

class Parameters():
    def __init__(self):     
        super(Parameters, self).__init__()
   
        
    def read_config(self, fn):
        config = configparser.ConfigParser()
        config.read(fn)
        
        self.read_weight_path = config.get("Record","read_weight_path")
        self.write_weight_folder=config.get("Record","write_weight_folder")
        self.write_bvh_motion_folder=config.get("Record","write_bvh_motion_folder")
        logdir = config.get("Record","logdir")
        self.logger = tensorboardX.SummaryWriter(logdir)
        
        self.discriminator_lst=[]
        record = config.items('Record')
        for (key, path) in record:
            if(len(key)>16):
                if(key[0:16]=="read_weight_disc"):
                    disc_id = key[16]
                    read_weight_disc_path = config.get('Record', key)
                    patch_len=int(config.get('Params','patch'+disc_id+"_len"))
                    patch_stride=int(config.get("Params", "patch"+disc_id+"_stride"))
                    patch_step=int(config.get("Params", "patch"+disc_id+"_step"))
                    self.discriminator_lst=self.discriminator_lst+[(disc_id, read_weight_disc_path, patch_len, patch_stride, patch_step)]
        
        
        self.lr =float( config.get("Params", "lr_ae"))
        
        self.target_frame_rate=int( config.get("Params", "target_frame_rate"))
       
        
        self.start_iteration=int(config.get("Params","start_iteration"))
        self.total_iteration=int( config.get("Params", "total_iteration"))
        self.save_bvh_iteration=int( config.get("Params", "save_bvh_iteration"))
        self.save_weight_iteration=int( config.get("Params", "save_weight_iteration"))
        
        self.out_rotation_mode = config.get("Params","out_rotation_mode")
        
        self.weight_pose=float( config.get("Params", "weight_pose"))
        self.weight_rotation_matrix=float( config.get("Params", "weight_rotation_matrix"))
        self.weight_twist=float(config.get("Params","weight_twist"))
        self.weight_pose_hip = float( config.get("Params", "weight_pose_hip"))
        
        self.loss_rmat_only_on_hip = int (config.get("Params", "loss_rmat_only_on_hip"))
        
        self.target_frame_rate=int(config.get("Params","target_frame_rate"))

        self.min_dance_seq_len = int(config.get("Params","min_dance_seq_len"))
        
        self.device = int(config.get("Params","device"))
        
        #self.global_motion = bool(int( config.get("Params", "global_motion")))
        
        
        dances=config.items("Dances")
        self.dances_folder_lst=[]
        for (key, path) in dances:
            if(key[len(key)-3:len(key)] != "_fr"):
                dance_folder = config.get("Dances", key)
                frame_rate=int(config.get("Dances", key+"_fr"))
                self.dances_folder_lst=self.dances_folder_lst+[(dance_folder, frame_rate)]
                
        
        







































