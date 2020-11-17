from typing import List, Any
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils.data_utils import Sequence

from config_v2 import Parameters
# from point_pillars import createPillars, createPillarsTarget
from point_pillars_v2 import createPillars, createPillarsTarget
# from readers import DataReader, Label3D
from sklearn.utils import shuffle
import sys

from det3d.pc_kitti_dataset import PCKittiAugmentedDataset

from point_viz.converter import PointvizConverter


def select_best_anchors(arr):
    dims = np.indices(arr.shape[1:])
    # arr[..., 0:1] gets the occupancy value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box}
    ind = (np.argmax(arr[..., 0:1], axis=0),) + tuple(dims)
    
    return arr[ind]


class DataProcessor(Parameters):

    def __init__(self, **kwargs):
        super(DataProcessor, self).__init__(**kwargs)
        anchor_dims = np.array(self.anchor_dims, dtype=np.float32)
        self.anchor_dims = anchor_dims[:, 0:3]
        self.anchor_z = anchor_dims[:, 3]
        self.anchor_yaw = anchor_dims[:, 4]
        # Counts may be used to make statistic about how well the anchor boxes fit the objects
        self.pos_cnt, self.neg_cnt = 0, 0

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

    def make_ground_truth(self, gt_boxes_3d: Any, gt_cls_type_list: List[str]):
        """ Generate the ground truth label for each pillars

        Args:
            gt_boxes_3d (numpy[float]): A list of floats containing [x, y, z, h, w, l, ry]
            gt_cls_type_list (List[str]): A list of floats containing [cls_type]

        Returns:
            [type]: [description]
        """

        # filter labels by classes (cars, pedestrians and Trams)
        # Label has 4 properties (Classification (0th index of labels file),
        # centroid coordinates, dimensions, yaw)
        # labels = list(filter(lambda x: x.classification in self.classes, labels))



        if len(gt_boxes_3d) == 0:
            pX, pY = int(self.Xn / self.downscaling_factor), int(self.Yn / self.downscaling_factor)
            a = int(self.anchor_dims.shape[0])
            return np.zeros((pX, pY, a), dtype='float32'), np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), np.zeros((pX, pY, a), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_classes), dtype='float64')

        # For each label file, generate these properties except for the Don't care class

        # target_positions = np.array([label.centroid for label in labels], dtype=np.float32)
        # target_dimension = np.array([label.dimension for label in labels], dtype=np.float32)
        # target_yaw = np.array([label.yaw for label in labels], dtype=np.float32)
        # target_class = np.array([self.classes[label.classification] for label in labels], dtype=np.int32)

        target_positions = gt_boxes_3d[:,:3]
        target_dimension = gt_boxes_3d[:,3:6] # don't have to translate again
        target_yaw = gt_boxes_3d[:, 6]
        # print(type(self.classes))
        # print(type(self.classes_map))
        # # print(gt_cls_type_list[0])
        # print(self.classes_map[gt_cls_type_list[0]])

        target_class = np.array([self.classes_map[gt_cls_type_list[k]] for k in range(len(gt_cls_type_list))], dtype=np.int32)

        assert np.all(target_yaw >= -np.pi) & np.all(target_yaw <= np.pi)
        assert len(target_positions) == len(target_dimension) == len(target_yaw) == len(target_class)

        target, pos, neg = createPillarsTarget(target_positions,
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
        self.pos_cnt += pos
        self.neg_cnt += neg

        # return a merged target view for all objects in the ground truth and get categorical labels
        # print("target.shape: ", target.shape)
        sel = select_best_anchors(target)
        ohe = tf.keras.utils.to_categorical(sel[..., 9], num_classes=self.nb_classes, dtype='float64')
        # print("self.shape: ", sel[...,0].shape)

        return sel[..., 0], sel[..., 1:4], sel[..., 4:7], sel[..., 7], sel[..., 8], ohe


class CustomDataGenerator(DataProcessor, Sequence, PCKittiAugmentedDataset):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, batch_size: int, root_dir:str, npoints:int =16384, split: str ='train', 
                classes:List[str] =['Car', 'Pedestrian', 'Person_sitting'], random_select:bool =True,
                gt_database_dir=None, aug_hard_ratio:float=0.5, **kwargs):

        super(CustomDataGenerator, self).__init__(
            batch_size=batch_size,  root_dir=root_dir, 
            npoints=npoints, split=split,   classes=classes, 
            random_select=random_select,    gt_database_dir=gt_database_dir, 
            aug_hard_ratio=aug_hard_ratio, **kwargs
        )
        # self.data_reader = data_reader
        self.batch_size = batch_size
        self.sample_id_list=self.get_sample_id_list()
        self.split = split


    def get_sample(self, index):
        return super().get_sample(index)


    def __len__(self):
        return len(self.sample_id_list) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = range(batch_id * self.batch_size, self.batch_size * (batch_id + 1))
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
            sample = self.get_sample(i)
            # For each file, dividing the space into a x-y grid to create pillars
            pts_lidar = sample['calib'].rect_to_lidar(sample['pts_rect'])
            pts_input = np.concatenate((pts_lidar, sample['pts_features']), axis=1)  # (N, C) 

            gt_boxes3d_xyz = sample['calib'].rect_to_lidar(sample['gt_boxes3d'][:,:3])

            gt_boxes3d = np.concatenate((
                gt_boxes3d_xyz[:,0,np.newaxis],   # 0 x
                gt_boxes3d_xyz[:,1,np.newaxis],   # 1 y
                gt_boxes3d_xyz[:,2,np.newaxis] + sample['gt_boxes3d'][:,3,np.newaxis] / 2,   # 2 z
                sample['gt_boxes3d'][:,5,np.newaxis],   # 3 l # same as the original label
                sample['gt_boxes3d'][:,4,np.newaxis],   # 4 w # same as the original label
                sample['gt_boxes3d'][:,3,np.newaxis],   # 5 h # same as the original label
                -sample['gt_boxes3d'][:,6,np.newaxis],   # 6 ry
            ), axis=1)

            # Voxels are the pillar ids
            pillars_, voxels_ = self.make_point_pillars(pts_input)

            pillars.append(pillars_)
            voxels.append(voxels_)

            
            if self.split=='train' or self.split =='val':
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(
                    gt_boxes3d, sample['gt_cls_type_list'])

                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)

        if self.split=='train' or self.split =='val':
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            return [pillars, voxels], [occupancy, position, size, angle, heading, classification]
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
        #         print("inside epoch")
        if self.split=='train' or self.split =='val':
            self.sample_id_list=shuffle(self.sample_id_list)


class AnalyseCustomDataGenerator(CustomDataGenerator):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, batch_size: int, root_dir:str, npoints:int =16384, split: str ='train', 
                classes:List[str] =['Car', 'Pedestrian', 'Person_sitting'], random_select:bool =True,
                gt_database_dir=None, aug_hard_ratio:float=0.5, **kwargs):

        super(AnalyseCustomDataGenerator, self).__init__(
            batch_size=batch_size,  root_dir=root_dir, 
            npoints=npoints, split=split,   classes=classes, 
            random_select=random_select,    gt_database_dir=gt_database_dir, 
            aug_hard_ratio=aug_hard_ratio, **kwargs
        )
        # self.data_reader = data_reader
        self.batch_size = batch_size
        self.sample_id_list=self.get_sample_id_list()
        self.split = split


    def get_sample(self, index):
        return super().get_sample(index)



    # def convert_labels_into_point_viz_format(self, gt_boxes3d):
    #     gt_boxes3d = gt_boxes3d[:,[3,4,5,0,1,2, 6]] # [xyz,3l4w5h,ry] => [3l,5h,4w]
    #     gt_boxes3d[:,5] -= (gt_boxes3d[:,2] /2)
    #     return gt_boxes3d

    def __len__(self):
        return len(self.sample_id_list) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = range(batch_id * self.batch_size, self.batch_size * (batch_id + 1))
        #         print("inside getitem")
        pillars = []
        voxels = []
        occupancy = []
        position = []
        size = []
        angle = []
        heading = []
        classification = []

        pts_input_ = []
        gt_boxes3d_ = []
        sample_ = []

        # save_viz_path = "/home/tan/tjtanaa/PointPillars/visualization/custom_processor"
        # # Initialize and setup output directory.
        # Converter = PointvizConverter(save_viz_path)

        for i in file_ids:
            # print(i)
            # print(type(i))
            sample = self.get_sample(i)
            # For each file, dividing the space into a x-y grid to create pillars
            pts_lidar = sample['calib'].rect_to_lidar(sample['pts_rect'])
            # print(pts_lidar.shape)

            pts_input = np.concatenate((pts_lidar, sample['pts_features']), axis=1)  # (N, C) 

            gt_boxes3d_xyz = sample['calib'].rect_to_lidar(sample['gt_boxes3d'][:,:3])

            # print(gt_boxes3d_xyz.shape)
            
            gt_boxes3d = np.concatenate((
                gt_boxes3d_xyz[:,0,np.newaxis],   # 0 x
                gt_boxes3d_xyz[:,1,np.newaxis],   # 1 y
                gt_boxes3d_xyz[:,2,np.newaxis] + sample['gt_boxes3d'][:,3,np.newaxis] / 2,   # 2 z
                sample['gt_boxes3d'][:,5,np.newaxis],   # 3 l # same as the original label
                sample['gt_boxes3d'][:,4,np.newaxis],   # 4 w # same as the original label
                sample['gt_boxes3d'][:,3,np.newaxis],   # 5 h # same as the original label
                -sample['gt_boxes3d'][:,6,np.newaxis],   # 6 ry
            ), axis=1)

            # print(type(gt_boxes3d))
            # gt_boxes3d = self.limit_yaw(gt_boxes3d)

            # bbox_params = self.convert_labels_into_point_viz_format(gt_boxes3d)
            # print(bbox_params.shape)
            # Converter.compile("custom_sample_{}".format(i), coors=pts_input[:,:3], intensity=pts_input[:,3],
            #                 bbox_params=bbox_params)
                
            
            # exit()

            # print(pts_input.shape)
            # Voxels are the pillar ids
            pillars_, voxels_ = self.make_point_pillars(pts_input)

            print(pillars_.shape, voxels_.shape)
            # for i in range(10):
            #     print(pillars_[0,0,i,:])
            # print(np.sum(pillars_ > 0))
            # exit()

            pillars.append(pillars_)
            voxels.append(voxels_)

            # print(sample['gt_cls_type_list'])
            
            if self.split=='train' or self.split =='val':
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(
                    gt_boxes3d, sample['gt_cls_type_list'])

                # print(occupancy_.shape, position_.shape, size_.shape, angle_.shape, heading_.shape, classification_.shape)


                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

                sample_.append(sample)
                gt_boxes3d_.append(gt_boxes3d)
                pts_input_.append(pts_input)

            # exit()

        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)

        if self.split=='train' or self.split =='val':
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            return [pillars, voxels], [occupancy, position, size, angle, heading, classification], [pts_input_, gt_boxes3d_, sample_]
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
        #         print("inside epoch")
        if self.split=='train' or self.split =='val':
            self.sample_id_list=shuffle(self.sample_id_list)
            