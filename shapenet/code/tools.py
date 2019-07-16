import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


    
#rotation5d batch*5
def normalize_5d_rotation( r5d):
    batch = r5d.shape[0]
    sin_cos = r5d[:,0:2] #batch*2
    sin_cos_mag = torch.max(torch.sqrt( sin_cos.pow(2).sum(1)), torch.autograd.Variable(torch.DoubleTensor([1e-8]).cuda()) ) #batch
    sin_cos_mag=sin_cos_mag.view(batch,1).expand(batch,2) #batch*2
    sin_cos = sin_cos/sin_cos_mag #batch*2
        
    axis = r5d[:,2:5] #batch*3
    axis_mag = torch.max(torch.sqrt( axis.pow(2).sum(1)), torch.autograd.Variable(torch.DoubleTensor([1e-8]).cuda()) ) #batch
        
    axis_mag=axis_mag.view(batch,1).expand(batch,3) #batch*3
    axis = axis/axis_mag #batch*3
    out_rotation = torch.cat((sin_cos, axis),1) #batch*5
    
    return out_rotation
    
#rotation5d batch*5
#out matrix batch*3*3
def rotation5d_to_matrix( r5d):
        
    batch = r5d.shape[0]
    sin = r5d[:,0].view(batch,1) #batch*1
    cos= r5d[:,1].view(batch,1) #batch*1
        
    x = r5d[:,2].view(batch,1) #batch*1
    y = r5d[:,3].view(batch,1) #batch*1
    z = r5d[:,4].view(batch,1) #batch*1
        
    row1 = torch.cat( (cos + x*x*(1-cos),  x*y*(1-cos)-z*sin, x*z*(1-cos)+y*sin ), 1) #batch*3
    row2 = torch.cat( (y*x*(1-cos)+z*sin,  cos+y*y*(1-cos),    y*z*(1-cos)-x*sin  ), 1) #batch*3
    row3 = torch.cat( (z*x*(1-cos)-y*sin,  z*y*(1-cos)+x*sin, cos+z*z*(1-cos)  ), 1) #batch*3
        
    matrix = torch.cat((row1.view(-1,1,3), row2.view(-1,1,3), row3.view(-1,1,3)), 1) #batch*3*3*seq_len
    matrix = matrix.view(batch, 3,3)
    return matrix
    
#T_poses num*3
#r_matrix batch*3*3
def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch=r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = r_matrix.view(batch,1, 3,3).expand(batch,joint_num, 3,3).contiguous().view(batch*joint_num,3,3)
    src_poses = T_pose.view(1,joint_num,3,1).expand(batch,joint_num,3,1).contiguous().view(batch*joint_num,3,1)
        
    out_poses = torch.matmul(r_matrices, src_poses) #(batch*joint_num)*3*1
        
    return out_poses.view(batch, joint_num,3)
    
# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

#u,a batch*3
#out batch*3
def proj_u_a(u,a):
    batch=u.shape[0]
    top = u[:,0]*a[:,0] + u[:,1]*a[:,1]+u[:,2]*a[:,2]
    bottom = u[:,0]*u[:,0] + u[:,1]*u[:,1]+u[:,2]*u[:,2]
    bottom = torch.max(torch.autograd.Variable(torch.zeros(batch).cuda())+1e-8, bottom)
    factor = (top/bottom).view(batch,1).expand(batch,3)
    out = factor* u
    return out

#matrices batch*3*3
def compute_rotation_matrix_from_matrix(matrices):
    b = matrices.shape[0]
    a1 = matrices[:,:,0]#batch*3
    a2 = matrices[:,:,1]
    a3 = matrices[:,:,2]
    
    u1 = a1
    u2 = a2 - proj_u_a(u1,a2)
    u3 = a3 - proj_u_a(u1,a3) - proj_u_a(u2,a3)
    
    e1 = normalize_vector(u1)
    e2 = normalize_vector(u2)
    e3 = normalize_vector(u3)
    
    rmat = torch.cat((e1.view(b, 3,1), e2.view(b,3,1),e3.view(b,3,1)), 2)
    
    return rmat
    
    
#in batch*5
#out batch*6
def stereographic_unproject_old(a):
    
    s2 = torch.pow(a,2).sum(1) #batch
    unproj= 2*a/ (s2+1).view(-1,1).repeat(1,5) #batch*5
    w = (s2-1)/(s2+1) #batch
    out = torch.cat((unproj, w.view(-1,1)), 1) #batch*6
    
    return out

#in a batch*5, axis int
def stereographic_unproject(a, axis=None):
    """
	Inverse of stereographic projection: increases dimension by one.
	"""
    batch=a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a,2).sum(1) #batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1]+1).cuda()) #batch*6
    unproj = 2*a/(s2+1).view(batch,1).repeat(1,a.shape[1]) #batch*5
    if(axis>0):
        ans[:,:axis] = unproj[:,:axis] #batch*(axis-0)
    ans[:,axis] = (s2-1)/(s2+1) #batch
    ans[:,axis+1:] = unproj[:,axis:]	 #batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used
    return ans



#a batch*5
#out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2)+1, np.sqrt(2)+1, np.sqrt(2)]) #3
    proj_scale = torch.autograd.Variable(torch.FloatTensor(proj_scale_np).cuda()).view(1,3).repeat(batch,1) #batch,3
    
    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)#batch*4
    norm = torch.sqrt(torch.pow(u[:,1:],2).sum(1)) #batch
    u = u/ norm.view(batch,1).repeat(1,u.shape[1]) #batch*4
    b = torch.cat((a[:,0:2], u),1)#batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix

#quaternion batch*4
def compute_rotation_matrix_from_quaternion( quaternion):
    batch=quaternion.shape[0]
    
    quat = normalize_vector(quaternion)
    
    qw = quat[...,0].view(batch, 1)
    qx = quat[...,1].view(batch, 1)
    qy = quat[...,2].view(batch, 1)
    qz = quat[...,3].view(batch, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix
    
#axisAngle batch*4 angle, x,y,z
def compute_rotation_matrix_from_axisAngle( axisAngle):
    batch = axisAngle.shape[0]
    
    theta = torch.tanh(axisAngle[:,0])*np.pi #[-180, 180]
    sin = torch.sin(theta)
    axis = normalize_vector(axisAngle[:,1:4]) #batch*3
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*3 a,b,c
def compute_rotation_matrix_from_hopf( hopf):
    batch = hopf.shape[0]
    
    theta = (torch.tanh(hopf[:,0])+1.0)*np.pi/2.0 #[0, pi]
    phi   = (torch.tanh(hopf[:,1])+1.0)*np.pi     #[0,2pi)
    tao   = (torch.tanh(hopf[:,2])+1.0)*np.pi     #[0,2pi)
    
    qw = torch.cos(theta/2)*torch.cos(tao/2)
    qx = torch.cos(theta/2)*torch.sin(tao/2)
    qy = torch.sin(theta/2)*torch.cos(phi+tao/2)
    qz = torch.sin(theta/2)*torch.sin(phi+tao/2)
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix
    

#euler batch*4
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)  
def compute_rotation_matrix_from_euler(euler):
    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix

#m batch*3*3
#out batch*4*4
def get_44_rotation_matrix_from_33_rotation_matrix(m):
    batch = m.shape[0]
    
    row4 = torch.autograd.Variable(torch.zeros(batch, 1,3).cuda())
    
    m43 = torch.cat((m, row4),1)#batch*4,3
    
    col4 = torch.autograd.Variable(torch.zeros(batch,4,1).cuda())
    col4[:,3,0]=col4[:,3,0]+1
    
    out=torch.cat((m43, col4), 2) #batch*4*4
    
    return out
    
    

#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    
    return theta


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to pi batch
def compute_angle_from_r_matrices(m):
    
    batch=m.shape[0]
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    theta = torch.acos(cos)
    
    return theta
    
def get_sampled_rotation_matrices_by_quat(batch):
    #quat = torch.autograd.Variable(torch.rand(batch,4).cuda())
    quat = torch.autograd.Variable(torch.randn(batch, 4).cuda())
    matrix = compute_rotation_matrix_from_quaternion(quat)
    return matrix
    
def get_sampled_rotation_matrices_by_hpof(batch):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,1, batch)*np.pi).cuda()) #[0, pi]
    phi   =  torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,2,batch)*np.pi).cuda())      #[0,2pi)
    tao   = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,2,batch)*np.pi).cuda())      #[0,2pi)
    
    
    qw = torch.cos(theta/2)*torch.cos(tao/2)
    qx = torch.cos(theta/2)*torch.sin(tao/2)
    qy = torch.sin(theta/2)*torch.cos(phi+tao/2)
    qz = torch.sin(theta/2)*torch.sin(phi+tao/2)
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*3*3s angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle( batch):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(-1,1, batch)*np.pi).cuda()) #[0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.autograd.Variable(torch.randn(batch, 3).cuda())
    axis = normalize_vector(axis) #batch*3
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix    
    
    
#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch=rotation_matrices.shape[0]
    R=rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular= sy<1e-6
    singular=singular.float()
        
    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)
    z=torch.atan2(R[:,1,0],R[:,0,0])
    
    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)
    zs=R[:,1,0]*0
        
    out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda())
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular
        
    return out_euler

#input batch*4
#output batch*4
def compute_quaternions_from_axisAngles(self, axisAngles):
    w = torch.cos(axisAngles[:,0]/2)
    sin = torch.sin(axisAngles[:,0]/2)
    x = sin*axisAngles[:,1]
    y = sin*axisAngles[:,2]
    z = sin*axisAngles[:,3]
    
    quat = torch.cat((w.view(-1,1), x.view(-1,1), y.view(-1,1), z.view(-1,1)), 1)
    
    return quat

#quaternions batch*4, 
#matrices batch*4*4 or batch*3*3
def compute_quaternions_from_rotation_matrices(matrices):
    batch=matrices.shape[0]
    
    w=torch.sqrt(1.0 + matrices[:,0,0] + matrices[:,1,1] + matrices[:,2,2]) / 2.0
    w = torch.max (w , torch.autograd.Variable(torch.zeros(batch).cuda())+1e-8) #batch
    w4 = 4.0 * w;
    x= (matrices[:,2,1] - matrices[:,1,2]) / w4 ;
    y= (matrices[:,0,2] - matrices[:,2,0]) / w4 ;
    z= (matrices[:,1,0] - matrices[:,0,1]) / w4 ;
        
    quats = torch.cat( (w.view(batch,1), x.view(batch, 1),y.view(batch, 1), z.view(batch, 1) ), 1   )
        
    return quats
    
    
    
