import os
from glob import glob
import numpy as np
import tensorflow as tf
from processors import SimpleDataGenerator, AnalyseSimpleDataGenerator
from inference_utils import generate_bboxes_from_pred, GroundTruthGenerator, focal_loss_checker, rotational_nms
from inference_utils import generate_bboxes_from_pred_and_np_array
from readers import KittiDataReader
from config import Parameters
from network import build_point_pillar_graph
from inference_utils import inverse_yaw_element

from point_viz.converter import PointvizConverter

DATA_ROOT = "/media/data3/tjtanaa/kitti_dataset/KITTI/object/training"
# MODEL_ROOT = "./logs_Car_Pedestrian_Original_2"
MODEL_ROOT = "./logs"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    # save_viz_path = "/home/tan/tjtanaa/PointPillars/visualization/custom_prediction_multiprocessing"
    save_viz_path = "/home/tan/tjtanaa/PointPillars/visualization/prediction"
    # Initialize and setup output directory.
    Converter = PointvizConverter(save_viz_path)

    params = Parameters()
    pillar_net = build_point_pillar_graph(params)
    pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
    # pillar_net.summary()

    data_reader = KittiDataReader()

    lidar_files = sorted(glob(os.path.join(DATA_ROOT, "velodyne", "*.bin")))[:100]
    print(len(lidar_files))
    print()
    label_files = sorted(glob(os.path.join(DATA_ROOT, "label_2", "*.txt")))[:100]
    calibration_files = sorted(glob(os.path.join(DATA_ROOT, "calib", "*.txt")))[:100]
    assert len(lidar_files) == len(label_files) == len(calibration_files), "Input dirs require equal number of files."
    eval_gen = AnalyseSimpleDataGenerator(data_reader, params.batch_size, lidar_files, label_files, calibration_files)


    for batch_idx in range(0,10):
        [pillars, voxels], [occupancy, position, size, angle, heading, classification], [pts_input, gt_boxes3d] \
            = eval_gen[batch_idx]

        # exit()

        occupancy, position, size, angle, heading, classification = pillar_net.predict([pillars, voxels])
        set_boxes, confidences = [], []
        loop_range = occupancy.shape[0] if len(occupancy.shape) == 4 else 1
        for i in range(loop_range):
            set_box, prediction = generate_bboxes_from_pred_and_np_array(occupancy[i], position[i], size[i], angle[i], heading[i],
                                                    classification[i], params.anchor_dims, occ_threshold=0.3)
            
            if len(set_box) == 0:
                continue
            set_boxes.append(set_box)
            # set_boxes.append(generate_bboxes_from_pred(occupancy[i], position[i], size[i], angle[i], heading[i],
                                                    #    classification[i], params.anchor_dims, occ_threshold=0.3))
            confidences.append([float(boxes.conf) for boxes in set_boxes[-1]])

            # print(set_boxes[0])
            # print(np.array(set_boxes[0]).shape)
            # print(prediction.shape)
            gt_boxes3d_ = []
            for j in range(len(gt_boxes3d[i])):
                bbox = gt_boxes3d[i][j]
                gt_boxes3d_.append([bbox.dimension[1], bbox.dimension[2], bbox.dimension[0],
                                    bbox.centroid[1], bbox.centroid[2] + bbox.dimension[2]/2, bbox.centroid[0]
                                    , -bbox.yaw])
            gt_boxes3d_np = np.array(gt_boxes3d_)
            print(gt_boxes3d_np.shape)

            Converter.compile("eval_sample_{}".format(batch_idx*params.batch_size + i), coors=pts_input[i][:,[1,2,0]], intensity=pts_input[i][:,3],
                        bbox_params=gt_boxes3d_np)
                        # bbox_params=gt_boxes3d_np[:,[3,5,4,1,2,0,6]])
    # print('Scene 1: Box predictions with occupancy > occ_thr: ', len(set_boxes[0]))

    # NMS
    # nms_boxes = rotational_nms(set_boxes, confidences, occ_threshold=0.3, nms_iou_thr=0.5)

    # print('Scene 1: Boxes after NMS with iou_thr: ', len(nms_boxes[0]))



    # # Do all the further operations on predicted_boxes array, which contains the predicted bounding boxes
    # gt_gen = GroundTruthGenerator(data_reader, label_files, calibration_files, network_format=False)
    # gt_gen0 = GroundTruthGenerator(data_reader, label_files, calibration_files, network_format=True)
    # for seq_boxes, gt_label, gt0 in zip(nms_boxes, gt_gen, gt_gen0):
    #     print("---------- New Scenario ---------- ")
    #     focal_loss_checker(gt0[0], occupancy[0], n_occs=-1)
    #     print("---------- ------------ ---------- ")
    #     for gt in gt_label:
    #         print(gt)
    #     for pred in seq_boxes:
    #         print(pred)
