Pytorch Code for "On The Continuity of Rotation Representations in Neural Networks".

## Environment

```
conda create -n env_Rotation python=3.6
conda activate env_Rotation
conda install numpy
pip install tensorboard tensorboardX transforms3d opencv-python matplotlib configparser scipy plyfile
conda install pytorch=0.4.1 cuda90 -c pytorch

```

## 1. Sanity Test

### Training
Go to sanity_test/code and run 

```
python train.py

```

This code will train the network with eight rotation representations using l2 loss on the rotation matrices or the geodesic loss. As defined in line 171, the directory of the output weights will be in ../train/test0000

### Testing
After the training is done, run 
```
python test.py

```

This code will print out the mean, max and std of the errors of different rotation representations.


## 2. Inverse Kinematics

### dataset
The dataset was constructed from a subset of CMU Motion Capture Dataset https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release

We transformed the original motion clips in the bvh form into npy form which contains a L*(3+57*4) array. L is the length of the sequence. 57 is the number of joints. the first three dimensions are the x,y,z position of the root(hip), the rest are the quaternions of the joints in the order defined in the bvh files.

Please download our transformed numpy training and testing data and put them in the folder Inverse_Kinematics/data.
https://drive.google.com/open?id=1ksAvPXoz-NwxikWIXS4SrlRtQyAbh7v8

https://drive.google.com/open?id=14cXYy_an6Lp1y0g8S8kJwEdlPykyQxVv

The code for transforming bvh to npy is in Inverse_Kinematics/code/generate_quat_data_from_bvh_data.py

For npy to bvh, find it in Inverse_Kinematics/code/read_bvh.write_joint_rotation_matrices_to_bvh

### Training
Run 
```
python trainIK.py

```
This code will load the configure file located in line 243 and train the network with the rotation representations indicated in the configure file. After running, the program will create the folders for writing the weights, temporal results in the form of bvh motion files and the log for tensorboard. The output directories and other parameters are defined in the configure file. I put six configure files for six rotation representations in the Inverse_Kinematics/training folder


### Testing
Run 
```
python trainIK_test.py

```

This code will load the models as defined in line 215 at iteration (line 222) and print out the mean and max position errors of the joints and the rotation errors of the root. It will also create folders in the folder of each model named as "/test_"+"%07d"%iteration" and output the bad poses with errors higher than a threshold (line 224) in bvh format.


## 3. ShapeNet
The data and code are organized in the way similar to the ones above.
















