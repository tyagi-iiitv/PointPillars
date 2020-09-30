import os
from glob import glob
import numpy as np
import tensorflow as tf
from processors import SimpleDataGenerator
from inference_utils import generate_bboxes_from_pred, GroundTruthGenerator, focal_loss_checker, rotational_nms
from readers import KittiDataReader
from config import Parameters
from network import build_point_pillar_graph

DATA_ROOT = "../training"
MODEL_ROOT = "./logs"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    params = Parameters()
    pillar_net = build_point_pillar_graph(params)
    pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
    # pillar_net.summary()

    data_reader = KittiDataReader()

    lidar_files = sorted(glob(os.path.join(DATA_ROOT, "velodyne", "*.bin")))
    label_files = sorted(glob(os.path.join(DATA_ROOT, "label_2", "*.txt")))
    calibration_files = sorted(glob(os.path.join(DATA_ROOT, "calib", "*.txt")))
    assert len(lidar_files) == len(label_files) == len(calibration_files), "Input dirs require equal number of files."
    eval_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files, label_files, calibration_files)

    occupancy, position, size, angle, heading, classification = pillar_net.predict(eval_gen,
                                                                                   batch_size=params.batch_size)
    set_boxes, confidences = [], []
    loop_range = occupancy.shape[0] if len(occupancy.shape) == 4 else 1
    for i in range(loop_range):
        set_boxes.append(generate_bboxes_from_pred(occupancy[i], position[i], size[i], angle[i], heading[i],
                                                   classification[i], params.anchor_dims, occ_threshold=0.7))
        confidences.append([float(boxes.conf) for boxes in set_boxes[-1]])
    print('Scene 1: Box predictions with occupancy > occ_thr: ', len(set_boxes[0]))

    # NMS
    nms_boxes = rotational_nms(set_boxes, confidences, occ_threshold=0.7, nms_iou_thr=0.5)

    print('Scene 1: Boxes after NMS with iou_thr: ', len(nms_boxes[0]))

    # Do all the further operations on predicted_boxes array, which contains the predicted bounding boxes
    gt_gen = GroundTruthGenerator(data_reader, label_files, calibration_files, network_format=False)
    gt_gen0 = GroundTruthGenerator(data_reader, label_files, calibration_files, network_format=True)
    for seq_boxes, gt_label, gt0 in zip(nms_boxes, gt_gen, gt_gen0):
        print("---------- New Scenario ---------- ")
        focal_loss_checker(gt0[0], occupancy[0], n_occs=-1)
        print("---------- ------------ ---------- ")
        for gt in gt_label:
            print(gt)
        for pred in seq_boxes:
            print(pred)
