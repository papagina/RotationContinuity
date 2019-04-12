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
        logdir = config.get("Record","logdir")
        self.logger = tensorboardX.SummaryWriter(logdir)
        
        
        self.lr =float( config.get("Params", "lr"))
        
        self.start_iteration=int(config.get("Params","start_iteration"))
        self.total_iteration=int( config.get("Params", "total_iteration"))
        self.save_weight_iteration=int( config.get("Params", "save_weight_iteration"))
        
        self.out_rotation_mode = config.get("Params","out_rotation_mode")
        
        #self.weight_pose=float( config.get("Params", "weight_pose"))
        self.weight_rmat=float( config.get("Params", "weight_rmat"))
        #self.weight_geodesic=float( config.get("Params", "weight_geodesic"))
        self.weight_translation=float(config.get("Params", "weight_translation"))
        
        self.regress_t = int(config.get("Params", "regress_t"))
        
        
        self.device = int(config.get("Params","device"))
        
        self.batch = int (config.get("Params","batch"))
        
        #self.global_motion = bool(int( config.get("Params", "global_motion")))
        
        
        shapes=config.items("Shapes")
        self.shape_folder_lst=[]
        for (key, path) in shapes:
            shape_folder = config.get("Shapes", key)
            self.shape_folder_lst=self.shape_folder_lst+[shape_folder]
                
        
        







































