

import os
from glob import glob
import numpy as np
import tensorflow as tf
from point_pillars_custom_processors_v2_2 import CustomDataGenerator, AnalyseCustomDataGenerator
from inference_utils_v2 import generate_bboxes_from_pred, GroundTruthGenerator
from inference_utils_v2 import focal_loss_checker, rotational_nms, generate_bboxes_from_pred_and_np_array, convert_boxes_to_list
from readers import KittiDataReader
from config_v2_2 import Parameters
from network_v2_2 import build_point_pillar_graph
from datetime import datetime

from det3d.kitti_dataset.utils.evaluation import save_kitti_format, save_kitti_format_for_evaluation

from point_viz.converter import PointvizConverter

DATA_ROOT = "/media/data3/tjtanaa/kitti_dataset/"
MODEL_ROOT = "./logs_Car_Custom_Dataset_No_Early_Stopping_wo_Aug_wo_val_new_network"

KITTI_EVALUATION_OUTPUT = os.path.join(MODEL_ROOT, "Evaluation")
if not os.path.exists(KITTI_EVALUATION_OUTPUT):
    os.makedirs(KITTI_EVALUATION_OUTPUT)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

if __name__ == "__main__":

    params = Parameters()
    pillar_net = build_point_pillar_graph(params)
    pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
    pillar_net.summary()

    # save_viz_path = "/home/tan/tjtanaa/PointPillars/visualization/custom_prediction_multiprocessing"
    save_viz_path = os.path.join("/home/tan/tjtanaa/PointPillars/visualization", MODEL_ROOT.split('/')[-1])
    # Initialize and setup output directory.
    Converter = PointvizConverter(save_viz_path)

    gt_database_dir = os.path.join(DATA_ROOT, "gt_database")

    validation_gen = AnalyseCustomDataGenerator(batch_size=params.batch_size,  root_dir=DATA_ROOT, 
            npoints=20000, split='train_val_test',random_select=False,  classes=list(params.classes_map.keys()))
    inference_duration = []
    sample_index = 0 # has to be controlled manually to ensure that the sequence number is continuous

    for batch_idx in range(0,len(validation_gen)):
        [pillars, voxels], [occupancy_, position_, size_, angle_, heading_], [pts_input, gt_boxes3d, sample] = validation_gen[batch_idx]

        start=datetime.now()

        occupancy, position, size, angle, heading = pillar_net.predict([pillars, voxels])

        inference_duration.append( datetime.now()-start)

        classification = np.zeros(shape=np.array(occupancy).shape)
        classification_ = classification

        set_boxes, confidences = [], []
        loop_range = occupancy_.shape[0] if len(occupancy_.shape) == 4 else 1
        for i in range(loop_range):
            set_box, predicted_boxes3d = generate_bboxes_from_pred_and_np_array(occupancy[i], position[i], size[i], angle[i], 
                                                            heading[i],
                                                            classification[i], params.anchor_dims, occ_threshold=0.5)


            _, decoded_gt_boxes3d = generate_bboxes_from_pred_and_np_array(occupancy_[i], position_[i], size_[i], angle_[i], 
                                                            heading_[i],
                                                            classification_[i], params.anchor_dims, occ_threshold=0.4)

            # gt_boxes3d_ = gt_boxes3d[i]
            gt_boxes3d_ = decoded_gt_boxes3d

            print(gt_boxes3d_.shape)
            if(len(gt_boxes3d_) == 0):
                gt_bbox_params_list = []
            else:
                gt_bbox_params = np.stack([gt_boxes3d_[:,3], gt_boxes3d_[:,5], gt_boxes3d_[:,4],
                            gt_boxes3d_[:,1], gt_boxes3d_[:,2] , 
                            gt_boxes3d_[:,0],
                            gt_boxes3d_[:,6]], axis=1)


                gt_bbox_params_list = gt_bbox_params.tolist()
            # gt_bbox_params_list = []
            for k in range(len(gt_bbox_params_list)):
                msg = "%.5f, %s, %.5f"%(decoded_gt_boxes3d[k,9], params.map_classes[int(decoded_gt_boxes3d[k,8])], decoded_gt_boxes3d[k,6])
                # msg = "%.5f, %.5f"%(gt_bbox_params_list[k][3],gt_bbox_params_list[k][5])
                gt_bbox_params_list[k].append("Green")
                # gt_bbox_params_list[k].append("1.0")
                gt_bbox_params_list[k].append(msg)

            if len(set_box) > 0:

                
                # NMS
                # set_box
                # print("start nms")
                confidence = [float(box.conf) for box in set_box]
                nms_boxes = rotational_nms([set_box], [confidence], occ_threshold=0.5, nms_iou_thr=0.5)

                predicted_boxes3d_list = convert_boxes_to_list(nms_boxes)

                predicted_boxes3d = np.array(predicted_boxes3d_list[0])
                predicted_boxes3d_ = predicted_boxes3d

                print("batch_idx: ", batch_idx * params.batch_size + i, " has ", predicted_boxes3d_.shape, "predictions")

                bbox_params = np.stack([predicted_boxes3d_[:,3], predicted_boxes3d_[:,5], predicted_boxes3d_[:,4],
                            predicted_boxes3d_[:,1], predicted_boxes3d_[:,2] , 
                            predicted_boxes3d_[:,0],
                            predicted_boxes3d_[:,6]], axis=1)


                bbox_params_list = bbox_params.tolist()
                # bbox_labels_conf = [str(predicted_boxes3d[k,9]) for k in range(predicted_boxes3d.shape[0])]
                for k in range(predicted_boxes3d.shape[0]):
                    msg = "%.5f, %s, %.5f"%(predicted_boxes3d[k,9],params.map_classes[int(predicted_boxes3d[k,8])], predicted_boxes3d[k,6])
                    bbox_params_list[k].append("Magenta")
                    bbox_params_list[k].append(msg)
                    # bbox_params_list[k].append(str(predicted_boxes3d[k,9]) + "=" + params.map_classes[int(predicted_boxes3d[k,8])])
                    gt_bbox_params_list.append(bbox_params_list[k])

                
                # save as kitti format for evaluation
                cur_sample_id = batch_idx * params.batch_size + i
                sample_file_name = validation_gen.sample_id_list[cur_sample_id]
                calib = sample[i]['calib']
                # cur_boxes3d = cur_boxes3d.cpu().numpy()
                
                cur_boxes3d_xyz = calib.lidar_to_rect(predicted_boxes3d[:, 0:3])

                cur_boxes3d = np.concatenate((
                    cur_boxes3d_xyz[:,0,np.newaxis],   # 0 x
                    cur_boxes3d_xyz[:,1,np.newaxis] + predicted_boxes3d[:,5,np.newaxis] / 2,   # 1 y
                    cur_boxes3d_xyz[:,2,np.newaxis],    # 2 z
                    predicted_boxes3d[:,5,np.newaxis],   # 3 l # same as the original label
                    predicted_boxes3d[:,4,np.newaxis],   # 4 w # same as the original label
                    predicted_boxes3d[:,3,np.newaxis],   # 5 h # same as the original label
                    -predicted_boxes3d[:,6,np.newaxis],   # 6 ry
                ), axis=1)
                cur_scores_raw = predicted_boxes3d[:,-1]
                image_shape = validation_gen.get_image_shape(sample_file_name)
                labels_obj = validation_gen.get_label(sample_file_name)
                classes = ['Car' for i in range(len(predicted_boxes3d))]
                save_kitti_format_for_evaluation(sample_index, calib, cur_boxes3d, KITTI_EVALUATION_OUTPUT, cur_scores_raw, image_shape, classes, labels_obj)
                sample_index += 1

            coor = pts_input[i][:,[1,2,0]]
            Converter.compile("evaluation_sample_{}".format(batch_idx * params.batch_size+i), coors=coor, intensity=pts_input[i][:,3],
                            bbox_params=gt_bbox_params_list)
    # print("Average runtime speed: ", np.mean(inference_duration[20:]))

