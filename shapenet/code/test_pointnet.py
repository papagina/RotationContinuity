import torch
import torch.nn as nn
import numpy as np
import tools
import Model_pointnet
import pts_loader
import train_pointnet_params as Params
import matplotlib.pyplot as plt
import os
import ntpath
from matplotlib.ticker import NullFormatter

from scipy import stats

def vector2string(data):
    s=' '.join(map(str, data))
    
    return s

def vectors2string(data):
    s='\n'.join(map(vector2string, data))
   
    return s

#numpy num*3
def save_point_clouds(pc1, pc2, fn):
    point_num =pc1.shape[0]
    red = np.array([[1,0,0]], dtype=int).repeat(point_num,0)#point_num*3
    pc1_red = np.append(pc1, red, axis = 1) #point_num*6
    
    blue = np.array([[0,1,0]], dtype=int).repeat(point_num,0)#point_num*3
    pc2_blue = np.append(pc2, blue, axis = 1) #point_num*6
    
    pc = np.append(pc1_red,pc2_blue, axis=0) #(2*point_num)*6
    file = open(fn, "w")
    data_str=vectors2string(pc)
    file.write(data_str)    
    file.close()
    

#input [folder_name]
#output [point_cloud]
#point_cloud num*3
def load_pc_fn_lst(pc_folder_lst):
    pc_fn_lst =[]
    for pc_folder in pc_folder_lst:
        pc_names = os.listdir(pc_folder)
        for pc_name in pc_names:
            pc_fn_lst = pc_fn_lst +[pc_folder+"/"+pc_name] 
            
    return pc_fn_lst
        
        
def test(pc_fn_lst, weight_fn,  out_rotation_mode, regress_t, rotation_sample_num ):#, save_out_pc_folder):
    torch.cuda.set_device(2)
    
    print ("####Initiate model")
    
    model = Model_pointnet.Model(out_rotation_mode=out_rotation_mode, regress_t=regress_t)
    
    print ("Load "+ weight_fn)
    model.load_state_dict(torch.load( weight_fn))
    model.cuda()
    model.eval()
    
    geodesic_errors_lst = np.array([])
    #rmat_errors_lst = np.array([])
    t_errors_lst = np.array([])
    num_big_geodesic_error =0
    for pc_fn in pc_fn_lst:
        batch=rotation_sample_num    
        pc1_np  = np.array(pts_loader.load(pc_fn))
        point_num = int(pc1_np.shape[0]/2)
        pc1 = torch.autograd.Variable(torch.FloatTensor(pc1_np[0:point_num]).cuda()) #num*3
        pc1 = pc1.view(1, point_num,3).expand(batch,point_num,3).contiguous() #batch*p_num*3
        gt_rmat = tools.get_sampled_rotation_matrices_by_axisAngle(batch)#batch*3*3
        gt_rmats = gt_rmat.contiguous().view(batch,1,3,3).expand(batch, point_num, 3,3 ).contiguous().view(-1,3,3)
        
        pc2 = torch.bmm(gt_rmats, pc1.view(-1,3,1))#(batch*point_num)*3*1
        pc2 = pc2.view(batch, point_num, 3) ##batch*p_num*3
        
        if(model.regress_t == True):
            gt_translation = torch.autograd.Variable(torch.FloatTensor(np.random.randn(batch,3)).cuda())/10.0
            pc2 = pc2 + gt_translation.view(batch,1,3).expand(batch, point_num, 3)
            ###network forward########
            out_rmat, out_translation, out_pc2 = model(pc1, pc2)#batch*3*3
        else:
            out_rmat, out_pc2 = model(pc1,pc2)
        
        geodesic_errors = np.array( tools.compute_geodesic_distance_from_two_matrices(gt_rmat, out_rmat).data.tolist()) #batch
        geodesic_errors = geodesic_errors/np.pi*180
        geodesic_errors_lst = np.append(geodesic_errors_lst, geodesic_errors)
        #rmat_errors = np.array(torch.pow(out_rmat - gt_rmat, 2).mean(2).mean(1).tolist()) #batch
        
        if(model.regress_t==True):
            t_errors = np.array(torch.sqrt( torch.pow(gt_translation - out_translation, 2).sum(1) ).data.tolist())
            t_errors_lst=np.append(t_errors_lst, t_errors)
     
    return geodesic_errors_lst, t_errors_lst
"""   
        if(geodesic_errors.max()>170):
            worst_id = np.argmax(geodesic_errors)
            head, tail = ntpath.split(pc_fn)
            print (tail)
            save_point_clouds(np.array(pc2[worst_id].data.tolist()), np.array(out_pc2[worst_id].data.tolist()), fn=save_out_pc_folder+tail[0:len(tail)-3]+"xyz")
        for error  in geodesic_errors:
            if(error>90):
                num_big_geodesic_error= num_big_geodesic_error+1
        
    
    avg_geodesic_error = geodesic_errors_lst.mean()
    max_geodesic_error = geodesic_errors_lst.max()
    print ("avg geodesic_error: " + str(avg_geodesic_error))
    print ("max geodesic_error: " + str(max_geodesic_error))   
    print (("big_geodesic_error_rate: ") + str(num_big_geodesic_error/len(geodesic_errors_lst)))
    
    if(model.regress_t==True):
        avg_t_error = t_errors_lst.mean()
        max_t_error = t_errors_lst.max()
        print ("avg t_error: " + str(avg_t_error))
        print ("max t_error: " + str(max_t_error))   
    return geodesic_errors_lst, t_errors_lst
"""
 
def visualize_error_lst(error_lst, histogram_bins_num, name):
    plt.hist(error_lst,  bins=histogram_bins_num, log=True)
    plt.ylabel(name)
    plt.legend(loc='upper right')
    plt.show()    
        
    plt.clf()


#test_pc_folder_lst = ["../pc_plane/points/"]


param=Params.Parameters()


#param.read_config("../train/test0301_plane_ortho6d/test0301_plane_ortho6d.config")
#param.read_config("../train/test0302_plane_rmat/test0302_plane_rmat.config")
#param.read_config("../train/test0303_plane_quat/test0303_plane_quat.config")
#param.read_config("../train/test0304_plane_axisAngle/test0304_plane_axisAngle.config")
#param.read_config("../train/test0305_plane_euler/test0305_plane_euler.config")
#param.read_config("../train/test0306_plane_ortho5d/test0306_plane_ortho5d.config")

#out_rotation_mode = param.out_rotation_mode
#regress_t = param.regress_t


#weight_fn ="../train/test0301_plane_ortho6d/weight/model_1900000.weight"
#weight_fn="../train/test0302_plane_rmat/weight/model_1900000.weight"
#weight_fn ="../train/test0303_plane_quat/weight/model_1160000.weight"
#weight_fn ="../train/test0304_plane_axisAngle/weight/model_1160000.weight"
#weight_fn ="../train/test0305_plane_euler/weight/model_1160000.weight"
#weight_fn ="../train/test0306_plane_ortho5d/weight/model_1160000.weight"


#save_out_pc_folder="../train/test0301_plane_ortho6d/test_1900k/"
#save_out_pc_folder="../train/test0302_plane_rmat/test_1900k/"
#save_out_pc_folder="../train/test0303_plane_quat/test_1160k/"
#save_out_pc_folder="../train/test0304_plane_axisAngle/test_1160k/"
#save_out_pc_folder="../train/test0305_plane_euler/test_1160k/"
#save_out_pc_folder="../train/test0306_plane_ortho5d/test_1160k/"

#if not os.path.exists(save_out_pc_folder):
#    os.makedirs(save_out_pc_folder)



#with torch.no_grad():     
#    geodesic_errors, t_errors = test(pc_fn_lst, weight_fn, out_rotation_mode, regress_t, rotation_sample_num=10)#,save_out_pc_folder=save_out_pc_folder)


#visualize_error_lst(geodesic_errors, histogram_bins_num=10, name = "geodesic errors")
#visualize_error_lst(t_errors, histogram_bins_num=10, name = "mse t errors")



def format_func_y(value, tick_number):
    percent = value*100
    if(percent>99.99):
        percent = 100
    elif(percent<0.000001):
        percent = 0
    out = "%.2f"%percent+"%"
    return out

def format_func_x(value, tick_number):
    out = "%d"%(int(value))+'°'
    return out

def get_error_lst(test_pc_fn_lst, weight_folder, model_name_lst,  iteration, sample_num):
    errors_lst=[]
    for (weight_sub_folder, out_rotation_mode) in model_name_lst:
        weight_fn  = weight_folder + weight_sub_folder+"/weight/model_%07d.weight"%iteration
        with torch.no_grad():
            geodesic_errors, t_errors = test(test_pc_fn_lst, weight_fn, out_rotation_mode, regress_t = 0, rotation_sample_num= sample_num)
        errors_lst = errors_lst + [(geodesic_errors, out_rotation_mode)]
    return errors_lst


test_pc_folder_lst = ["../pc_plane/points_test"]
pc_fn_lst= load_pc_fn_lst(test_pc_folder_lst) #[3*num]



weight_folder= "../train/"
model_name_lst = [ ("test0301_plane_ortho6d","ortho6d"),
                  ("test0306_plane_ortho5d","ortho5d"),
                   ("test0303_plane_quat","Quaternion"),
                    ("test0304_plane_axisAngle","axisAngle"),
                    ("test0305_plane_euler","euler"),
                    ("test0302_plane_rmat","rmat")]

iteration=2500000

#errors_lst = get_error_lst(pc_fn_lst, weight_folder,model_name_lst,  iteration, sample_num=10)

for errors, name in errors_lst:
    print (name)
    print ("avg:" + str( np.round( errors.mean(),2) ))
    print ("max:" + str( np.round(errors.max(),2) ))
    print ("std:"+  str( np.round(errors.std(),2) ))

def format_func_x_percentile_logit(value, tick_number):
    
    percent = value*100
    
    out = "%.2f"%percent+"%"
    return out


def format_func_x_percentile(value, tick_number):
    out = str(int(value))+"%"
    return out


def format_func_y_degree_logit(value, tick_number):
    
    score = value*180
    
    out = str(score)+"°"
    return out


def format_func_y_degree_log(value, tick_number):
    
    score = np.exp(value)
    
    out = str(int(score))+"°"
    return out

def format_func_y_degree(value,tick_number):
    out  = str(int(value))+"°"
    return out

for (errors, mode_name) in errors_lst:
    line_shape="-"
    if(mode_name =="axisAngle"):
        color = "c"
        plot_name = "Axis-angle"
    elif(mode_name =="euler"):
        color = "b"
        plot_name="Euler"
    elif(mode_name =="Quaternion" ):
        color = "g"
        plot_name="Quaternion"
    elif(mode_name =="ortho5d" ):
        color = "y"
        plot_name ="5D"
        
    elif(mode_name =="ortho6d"):
        color = "r"
        plot_name = "6D"
    elif(mode_name =="rmat"):
        color = "m"
        plot_name = "3X3 Matrix"

    percentile_lst = []
    score_lst = []
    i_lst = np.linspace(0, 100, num=400)
    for i in i_lst:
        percentile = i
        score = stats.scoreatpercentile(errors, percentile)
        score = np.log(score)
        #score= score/180
        #if(score==0):
        #    score = score-0.00000001
        #print (score)
        
            
        score_lst = score_lst + [score]
        percentile_lst = percentile_lst +[percentile]
    
    percentile_lst = np.array(percentile_lst)
    score_lst = np.array(score_lst)
    shape=""
    #line_shape=""
    #plt.plot(axis, percents, color+shape, label=name[0:len(name)-2])
    plt.plot(percentile_lst, score_lst, color+line_shape, label=plot_name)

#plt.yscale('logit')
    #pylab.yticks
yticks = np.array([0.1, 1, 10, 100, 180 ])
plt.yticks(np.log(yticks))
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree_log))
#plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_x_percentile))
plt.legend(loc='upper left')
plt.grid()

plt.savefig("error_percentile_0.pdf")
plt.show()
plt.clf()

"""

error_groups =[[0,30],[30,60],[60,90],[90,120],[120,150],[150,180]]
error_groups= []
for i in range(18):
    error_groups = error_groups + [[10*i,10*i+10]]
#error_groups = [[0,60],[60,120],[120,180]]
for (errors, mode_name) in errors_lst:
    line_shape="-"
    if(mode_name =="axisAngle"):
        color = "c"
        plot_name = "Axis-angle"
    elif(mode_name =="euler"):
        color = "b"
        plot_name="Euler"
    elif(mode_name =="Quaternion" ):
        color = "g"
        plot_name="Quaternion"
    elif(mode_name =="ortho5d" ):
        color = "y"
        plot_name ="5D"
        
    elif(mode_name =="ortho6d"):
        color = "r"
        plot_name = "6D"
    elif(mode_name =="rmat"):
        color = "m"
        plot_name = "3X3 Matrix"
        
    percents =[]
    axis = []
    for group in error_groups:
        num =0
        for error in errors:
            if((error<group[1]) and (error>=group[0])):
                num=num+1
        percent = num/errors.shape[0]
        if(percent == 1.0):
            percent = percent-1e-5
        elif(percent == 0.0):
            percent = percent+1e-5
        
        percents = percents+[percent]
        axis = axis + [group[1]-5]
    percents = np.array(percents)
    
    #percents=np.log(percents)
    axis = np.array(axis)
    
    plt.plot(axis, percents, color+ line_shape, label=plot_name)
    #plt.plot(axis, percents, color+"--")


plt.yscale('logit')
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y))
#plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_x))
plt.legend(loc='upper right')
plt.grid()

plt.savefig("error_distribution.pdf")
plt.show()
plt.clf()



def format_func_x2(value, tick_number):
    out = "%i"%(value/1000)+'k'
    return out

def format_func_y2(value, tick_number):
    out = "%d"%value+'°'
    return out


plt.clf()
weight_folder= "../train/"
iterations = np.linspace(200000, 2600000, num=16)
model_name_lst = [ ("test0301_plane_ortho6d","ortho6d"),
                  ("test0306_plane_ortho5d","ortho5d"),
                   ("test0303_plane_quat","Quaternion"),
                    ("test0304_plane_axisAngle","axisAngle"),
                    ("test0305_plane_euler","euler"),
                    ("test0302_plane_rmat","rmat")]
errors_lst_lst =[]
for i in iterations:
    errors_lst = get_error_lst(pc_fn_lst, weight_folder, model_name_lst,  i, sample_num=1)
    errors_lst_lst = errors_lst_lst + [(errors_lst, i)]

for m in range(len(model_name_lst)):
    errors_through_iter = []
    weight_sub_folder, mode_name = model_name_lst[m]
    
    if(mode_name =="axisAngle"):
        color = "c"
        plot_name = "Axis-angle"
    elif(mode_name =="euler"):
        color = "b"
        plot_name="Euler"
    elif(mode_name =="Quaternion" ):
        color = "g"
        plot_name="Quaternion"
    elif(mode_name =="ortho5d" ):
        color = "y"
        plot_name ="5D"
        
    elif(mode_name =="ortho6d"):
        color = "r"
        plot_name = "6D"
    elif(mode_name =="rmat"):
        color = "m"
        plot_name = "3X3 Matrix"
    for i in range(len(iterations)):
        error = errors_lst_lst[i][0][m][0].mean()
        errors_through_iter = errors_through_iter + [error]
    errors_through_iter = np.array(errors_through_iter)
    
    #print (name)
    #print ("avg error:" + str(errors_through_iter[len(errors_through_iter)-1]))
    
    plt.plot(iterations, errors_through_iter, color, label=plot_name)
    #plt.plot(iterations, errors_through_iter, color)


#plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y2))
#plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10000))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_x2))
plt.legend(loc='upper right')
plt.grid()

plt.savefig("avg_error_through_iter.pdf")
plt.show()
plt.clf()




"""












