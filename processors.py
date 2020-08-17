from typing import List
import numpy as np

from tensorflow.python.keras.utils.data_utils import Sequence

from config import Parameters
from point_pillars import createPillars, createPillarsTarget
from readers import DataReader, Label3D
from sklearn.utils import shuffle
import sys


def select(x, m):
    dims = np.indices(x.shape[1:])
    ind = (m,) + tuple(dims)
    return x[ind]


class DataProcessor(Parameters):

    def __init__(self):
        super(DataProcessor, self).__init__()
        anchor_dims = np.array(self.anchor_dims, dtype=np.float32)
        self.anchor_dims = anchor_dims[:, 0:3]
        self.anchor_z = anchor_dims[:, 3]
        self.anchor_yaw = anchor_dims[:, 4]

    @staticmethod
    def transform_labels_into_lidar_coordinates(labels: List[Label3D], R: np.ndarray, t: np.ndarray):
        transformed = []
        for label in labels:
            label.centroid = label.centroid @ np.linalg.inv(R).T - t
            label.dimension = label.dimension[[2, 1, 0]]
            label.yaw -= np.pi / 2
            while label.yaw < -np.pi:
                label.yaw += (np.pi * 2)
            while label.yaw > np.pi:
                label.yaw -= (np.pi * 2)
            transformed.append(label)
        return labels

    def make_point_pillars(self, points: np.ndarray):

        assert points.ndim == 2
        assert points.shape[1] == 4
        assert points.dtype == np.float32

        pillars, indices = createPillars(points,
                                         self.max_points_per_pillar,
                                         self.max_pillars,
                                         self.x_step,
                                         self.y_step,
                                         self.x_min,
                                         self.x_max,
                                         self.y_min,
                                         self.y_max,
                                         self.z_min,
                                         self.z_max,
                                         False)

        return pillars, indices

    def make_ground_truth(self, labels: List[Label3D]):

        # filter labels by classes (cars, pedestrians and Trams)
        # Label has 4 properties (Classification (0th index of labels file), 
        # centroid coordinates, dimensions, yaw)
        labels = list(filter(lambda x: x.classification in self.classes, labels))
        
        if len(labels) == 0:
            return
        
        
        #For each label file, generate these properties except for the Don't care class
        target_positions = np.array([label.centroid for label in labels], dtype=np.float32)
        target_dimension = np.array([label.dimension for label in labels], dtype=np.float32)
        target_yaw = np.array([label.yaw for label in labels], dtype=np.float32)
        target_class = np.array([self.classes[label.classification] for label in labels], dtype=np.int32)
        
        
        assert np.all(target_yaw >= -np.pi) & np.all(target_yaw <= np.pi)
        assert len(target_positions) == len(target_dimension) == len(target_yaw) == len(target_class)

        target = createPillarsTarget(target_positions,
                                     target_dimension,
                                     target_yaw,
                                     target_class,
                                     self.anchor_dims,
                                     self.anchor_z,
                                     self.anchor_yaw,
                                     self.positive_iou_threshold,
                                     self.negative_iou_threshold,
                                     self.nb_classes,
                                     self.downscaling_factor,
                                     self.x_step,
                                     self.y_step,
                                     self.x_min,
                                     self.x_max,
                                     self.y_min,
                                     self.y_max,
                                     self.z_min,
                                     self.z_max,
                                     False)

        print("Target shape", target.shape)
#         print(target[...,0:1].shape)
#         print(target)
#         sys.exit()
        # target[..., 0:1] gets the 0th value of 10 dim tensor for every (object,xgridcell,ygridcell,anchor)
        # We are trying to get the index of best anchors for each (object, xgridcell, ygridcell)
        best_anchors = target[..., 0:1].argmax(0) #This always returns zero. Verify that this is correct. 
#         print("Best Anchor shape", best_anchors.shape)
#         print(best_anchors)
        selection = select(target, best_anchors)
        print("Selection shape", selection.shape)
        sys.exit()
        # Selection is best anchor for each (object, xgridcell, ygridcell). See that this is a 10 dim vector containing info about the best anchor for each grid cell. 
        
        # one hot encoding of class
        clf = selection[..., 9] #9th vector contains the classification id
        # clf contains the class if of best anchor for each grid cell
        clf[clf == -1] = 0
        ohe = np.eye(self.nb_classes)[np.array(clf, dtype=np.int32).reshape(-1)]
        ohe = ohe.reshape(list(clf.shape) + [self.nb_classes])

        return selection[..., 0], selection[..., 1:4], selection[..., 4:7], selection[..., 7], selection[..., 8], ohe


class SimpleDataGenerator(DataProcessor, Sequence):

    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, data_reader: DataReader, batch_size: int, lidar_files: List[str], label_files: List[str] = None,
                 calibration_files: List[str] = None):
        super(SimpleDataGenerator, self).__init__()
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.lidar_files = lidar_files
        self.label_files = label_files
        self.calibration_files = calibration_files

        assert (calibration_files is None and label_files is None) or \
               (calibration_files is not None and label_files is not None)

        if self.calibration_files is not None:
            assert len(self.calibration_files) == len(self.lidar_files)
            assert len(self.label_files) == len(self.lidar_files)

    def __len__(self):
        return len(self.lidar_files) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = np.arange(batch_id * self.batch_size, self.batch_size * (batch_id + 1))
#         print("inside getitem")
        pillars = []
        voxels = []
        occupancy = []
        position = []
        size = []
        angle = []
        heading = []
        classification = []
        
        
        for i in file_ids:
            lidar = self.data_reader.read_lidar(self.lidar_files[i])
            # For each file, dividing the space into a x-y grid to create pillars
            # Voxels are the pillar ids
            pillars_, voxels_ = self.make_point_pillars(lidar)

            pillars.append(pillars_)
            voxels.append(voxels_)

            if self.label_files is not None:
                label = self.data_reader.read_label(self.label_files[i])
                R, t = self.data_reader.read_calibration(self.calibration_files[i])
                # Labels are transformed into the lidar coordinate bounding boxes
                # Label has 7 values, centroid, dimensions and yaw value. 
                label_transformed = self.transform_labels_into_lidar_coordinates(label, R, t)
                # These definitions can be found in point_pillars.cpp file
                # We are splitting a 10 dim vector that contains this information.
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(label_transformed)

                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)

        if self.label_files is not None:
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
#             print(pillars[0], pillars[0].shape)
#             print("..................")
#             print(voxels[0], voxels[0].shape)
#             print("..................")
#             print(occupancy[0], occupancy[0].shape)
#             print("..................")
#             print(position[0], position[0].shape)
#             print("..................")
#             print(size[0], size[0].shape)
#             print("..................")
#             print(angle[0], angle[0].shape)
#             print("..................")
#             print(heading[0], heading[0].shape)
#             print("..................")
#             print(classification[0], classification[0].shape)
#             sys.exit()
            '''
            Pillars (12000,100,7) for 12000 pillars, each pillar has 100 points (included zero padded points). And each point has 7 values (x,y,z,intensity,xc,yc,zc).
            
            Voxels (12000,3) The x,y,z center of each pillar.
            
            Occupancy, Position, Size, Angle, Heading, Classification are calculated from point_pillars.cpp file where we are splitting a 10 dim vector
            '''
            return [pillars, voxels], [occupancy, position, size, angle, heading, classification]
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
#         print("inside epoch")
#         shuffled_indices = np.random.permutation(np.arange(0, len(self.lidar_files)))
#         self.lidar_files = self.lidar_files[shuffled_indices]

        if self.label_files is not None:
            self.lidar_files, self.label_files, self.calibration_files = shuffle(self.lidar_files, self.label_files, self.calibration_files)
#             self.calibration_files = self.calibration_files[shuffled_indices]
#             self.label_files = self.label_files[shuffled_indices]
