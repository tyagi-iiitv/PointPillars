import os
import tensorflow as tf
from glob import glob
from processors import SimpleDataGenerator
from readers import KittiDataReader
from config import Parameters
from network import build_point_pillar_graph

DATA_ROOT = "../validation_small"  
MODEL_ROOT = "./logs"

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == "__main__":

    params = Parameters()
    pillar_net = build_point_pillar_graph(params)
    pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
    pillar_net.summary()

    data_reader = KittiDataReader()

    lidar_files = sorted(glob(os.path.join(DATA_ROOT, "velodyne", "*.bin")))
    label_files = sorted(glob(os.path.join(DATA_ROOT, "label_2", "*.txt")))
    calibration_files = sorted(glob(os.path.join(DATA_ROOT, "calib", "*.txt")))

    eval_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files, label_files, calibration_files)
    # eval_gen returns voxels and pillar_index for that bin file
#     print(eval_gen)
    occupancy, loc, size, angle, heading, clf = pillar_net.predict(eval_gen)
    
    
    
    