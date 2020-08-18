# PointPillars in TensorFlow
Point PIllars 3D detection network implementation in Tensorflow. External contributions are welcome, please fork this repo and see the issues for possible improvements in the code.  

# Installation
Download the LiDAR, Calibration and Label_2 **zip** files from the [Kitti dataset link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and unzip the files, giving the following directory structure:

```plain
├── training    <-- 7481 train data
   |   ├── calib
   |   ├── label_2
   |   ├── velodyne
└── testing     <-- 7580 test data
           ├── calib
           ├── velodyne
```
After placing the Kitti dataset in the root directory, run the following code

```
git clone --recurse-submodules https://github.com/tyagi-iiitv/PointPillars.git
virtualenv --python=/usr/bin/python3.8 env
source ./env/bin/activate
pip install tensorflow-gpu
pip install sklearn
cd PointPillars
python setup.py install
python point_pillars_training_run.py
```

# Deploy on a cloud notebook instance (Amazon SageMaker etc.)
Please read this blog article: https://link.medium.com/TVNzx03En8

# Technical details about this code
Please refer to this [article](https://medium.com/@a_tyagi/implementing-point-pillars-in-tensorflow-c38d10e9286?source=friends_link&sk=90995fae2d0a9c4e0dd5ec420c218c84) on Medium. 


