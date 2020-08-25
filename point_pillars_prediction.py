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
    
    occupancy, position, size, angle, heading, classification = pillar_net.predict(eval_gen, batch_size=1)
    
    # Generating the bounding boxes based on the regression targets
    
    # Get only the boxes where Occupancy is 1. 
    real_boxes = np.where(occupancy == 1)
    # Get the indices of the occupancy array where it is 1
    coordinates = list(zip(real_boxes[0], real_boxes[1], real_boxes[2]))
    # Assign anchor dimensions as original bounding box coordinates which will eventually be changed 
    # according to the predicted regression targets
    anchor_dims = params.anchor_dims
    real_anchors = np.random.rand(len(coordinates),len(anchor_dims[0]))

    for i,value in enumerate(real_boxes[2]):
        real_anchors[i,...] = anchor_dims[value]
    
    # Change the anchor boxes based on regression targets, this is the inverse of the operations given in 
    # createPillarTargets function (src/PointPillars.cpp)
    predicted_boxes = []
    for i,value in enumerate(coordinates):
        real_diag = np.sqrt(np.square(real_anchors[i][0]) + np.square(real_anchors[i][1]))
        real_x = value[0] * Parameters.x_step * Parameters.downscaling_factor + Parameters.x_min
        real_y = value[1] * Parameters.y_step * Parameters.downscaling_factor + Parameters.y_min
        bb_x = position[value][0] * real_diag + real_x
        bb_y = position[value][1] * real_diag + real_y
        bb_z = position[value][2] * real_anchors[i][2] + real_anchors[i][3]
    #     print(position[value], real_x, real_y, real_diag)
        bb_length = np.exp(size[value][0])*real_anchors[i][0]
        bb_width = np.exp(size[value][1])*real_anchors[i][1]
        bb_height = np.exp(size[value][2])*real_anchors[i][2]
        bb_yaw = np.arcsin(angle[value]) + real_anchors[i][4]
        bb_heading = heading[value]
        bb_cls = np.where(classification[value] == 1)[0][0]
        predicted_boxes.append([bb_height, bb_width, bb_length, bb_y, bb_z, bb_x, bb_yaw, bb_heading,bb_cls])
    
    # Do all the further operations on predicted_boxes array, which contains the predicted bounding boxes
    
    
    
    
