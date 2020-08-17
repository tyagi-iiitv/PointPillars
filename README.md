# PointPillars in TensorFlow
Point PIllars 3D detection network implementation in Tensorflow. External contributions are welcome, please fork this repo and see the issues for possible improvements in the code.  

Download the LiDAR, Calibration and Label_2 **zip** files from the [Kitti dataset link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and unzip the files, giving the following directory structure:

.
+-- training
    +-- calib
    +-- label_2
    +-- velodyne
+-- testing
    +-- calib
    +-- velodyne


# Installation
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

# Instructions to deploy on a cloud notebook instance (Amazon SageMaker etc.)
Please read this blog article: https://link.medium.com/TVNzx03En8



