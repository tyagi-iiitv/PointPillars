import numpy as np
from typing import List
from config import Parameters
from readers import DataReader
from processors import DataProcessor


class BBox(object):
    def __init__(self, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls):
        self.x = bb_x
        self.y = bb_y
        self.z = bb_z
        self.length = bb_length
        self.width = bb_width
        self.height = bb_height
        self.yaw = bb_yaw
        self.heading = bb_heading
        self.cls = bb_cls

    def __str__(self):
        return "BB | Cls: %s, x: %f, y: %f, l: %f, w: %f, yaw: %f" % (
            self.cls, self.x, self.y, self.length, self.width, self.yaw)


def generate_bboxes_from_pred(occ, pos, siz, ang, hdg, clf, anchor_dims, occ_threshold=0.5):
    """ Generating the bounding boxes based on the regression targets """

    # Get only the boxes where occupancy is greater or equal threshold.
    real_boxes = np.where(occ >= occ_threshold)
    # Get the indices of the occupancy array
    coordinates = list(zip(real_boxes[0], real_boxes[1], real_boxes[2]))
    # Assign anchor dimensions as original bounding box coordinates which will eventually be changed
    # according to the predicted regression targets
    anchor_dims = anchor_dims
    real_anchors = np.random.rand(len(coordinates), len(anchor_dims[0]))

    for i, value in enumerate(real_boxes[2]):
        real_anchors[i, ...] = anchor_dims[value]

    # Change the anchor boxes based on regression targets, this is the inverse of the operations given in
    # createPillarTargets function (src/PointPillars.cpp)
    predicted_boxes = []
    for i, value in enumerate(coordinates):
        real_diag = np.sqrt(np.square(real_anchors[i][0]) + np.square(real_anchors[i][1]))
        real_x = value[0] * Parameters.x_step * Parameters.downscaling_factor + Parameters.x_min
        real_y = value[1] * Parameters.y_step * Parameters.downscaling_factor + Parameters.y_min
        bb_x = pos[value][0] * real_diag + real_x
        bb_y = pos[value][1] * real_diag + real_y
        bb_z = pos[value][2] * real_anchors[i][2] + real_anchors[i][3]
        # print(position[value], real_x, real_y, real_diag)
        bb_length = np.exp(siz[value][0]) * real_anchors[i][0]
        bb_width = np.exp(siz[value][1]) * real_anchors[i][1]
        bb_height = np.exp(siz[value][2]) * real_anchors[i][2]
        bb_yaw = -np.arcsin(np.clip(ang[value], -1, 1)) + real_anchors[i][4]
        bb_heading = np.round(hdg[value])
        bb_cls = np.argmax(clf[value])
        bb_conf = occ[value]
        predicted_boxes.append(BBox(bb_x, bb_y, bb_z, bb_length, bb_width, bb_height,
                                    bb_yaw, bb_heading, bb_cls, bb_conf))

    return predicted_boxes


class GroundTruthGenerator(DataProcessor):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, data_reader: DataReader, label_files: List[str], calibration_files: List[str] = None):
        super(GroundTruthGenerator, self).__init__()
        self.data_reader = data_reader
        self.label_files = label_files
        self.calibration_files = calibration_files

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, file_id: int):
        label = self.data_reader.read_label(self.label_files[file_id])
        R, t = self.data_reader.read_calibration(self.calibration_files[file_id])
        label_transformed = self.transform_labels_into_lidar_coordinates(label, R, t)
        return label_transformed
