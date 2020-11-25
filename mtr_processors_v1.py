from typing import List, Any
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils.data_utils import Sequence

from config_mtr_v1 import Parameters
# from point_pillars import createPillars, createPillarsTarget
from point_pillars_v2 import createPillars, createPillarsTarget
# from readers import DataReader, Label3D
from sklearn.utils import shuffle
import sys
from det3d.mtr_dataset import MTRDatasetBase
from det3d.mtr_dataset.utils import mtr_utils

# from point_viz.converter import PointvizConverter
from datetime import datetime


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

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
        # start=datetime.now()
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
        # print("Create pillar takes : ", datetime.now()-start)

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

        if len(gt_boxes_3d) == 0:
            pX, pY = int(self.Xn / self.downscaling_factor), int(self.Yn / self.downscaling_factor)
            a = int(self.anchor_dims.shape[0])
            return np.zeros((pX, pY, a), dtype='float32'), np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), np.zeros((pX, pY, a), dtype='float32'), \
                   np.zeros((pX, pY, a), dtype='float32'), np.zeros((pX, pY, a, self.nb_classes), dtype='float64')

        # For each label file, generate these properties except for the Don't care class
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

        # start=datetime.now()

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
        
        # print("Create target takes : ", datetime.now()-start)

        self.pos_cnt += pos
        self.neg_cnt += neg

        # return a merged target view for all objects in the ground truth and get categorical labels

        sel = select_best_anchors(target)
        ohe = tf.keras.utils.to_categorical(sel[..., 9], num_classes=self.nb_classes, dtype='float64')
        # print("self.shape: ", sel[...,0].shape)


        return sel[..., 0], sel[..., 1:4], sel[..., 4:7], sel[..., 7], sel[..., 8], ohe


class CustomDataGenerator(DataProcessor, Sequence, MTRDatasetBase):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, batch_size: int, root_dir:str, point_cloud_statistics_path: str,
                npoints:int =16384, split: str ='train', 
                classes:List[str] =['Car', 'Pedestrian', 'Person_sitting'], random_select:bool =True,
                gt_database_dir=None, aug_hard_ratio:float=0.5, **kwargs):

        super(CustomDataGenerator, self).__init__(
            root_dir = root_dir,
            split = split,
            point_cloud_statistics_path = point_cloud_statistics_path,
            **kwargs
            # batch_size=batch_size,  root_dir=root_dir, 
            # npoints=npoints, split=split,   classes=classes, 
            # random_select=random_select,    gt_database_dir=gt_database_dir, 
            # aug_hard_ratio=aug_hard_ratio, **kwargs
        )
        self.batch_size = batch_size


    def get_sample(self, index):
        return super().get_sample(index)


    def __len__(self):
        return len(self.sample_list) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = range(batch_id * self.batch_size, self.batch_size * (batch_id + 1))
        pillars = []
        voxels = []
        occupancy = []
        position = []
        size = []
        angle = []
        heading = []
        classification = []



        for i in file_ids:
            point_cloud = self.get_lidar_without_background(i)

            pts_features = point_cloud[:, 3:]

            pts_input = np.concatenate([point_cloud[:,:3], pts_features[:,1,np.newaxis]], axis=1)

            # Voxels are the pillar ids
            pillars_, voxels_ = self.make_point_pillars(pts_input)

            pillars.append(pillars_)
            voxels.append(voxels_)



            obj_list = self.get_label(i) # are labels

            gt_boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
            gt_bbox_params_list = []
            for k, obj in enumerate(obj_list):
                gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, limit_period(obj.ry, offset=0.5, period=2*np.pi) # mtr format
        #             = obj.pos, obj.h, obj.w, obj.l, obj.ry # kitti

        #     print(bboxes3d_[:,:3].shape)
            invalid_region_mask = self._get_invalid_region_mask(gt_boxes3d[:,:3])
            gt_boxes3d = gt_boxes3d[~invalid_region_mask,:]
            
            gt_boxes3d = np.concatenate((
                gt_boxes3d[:,0,np.newaxis],   # 0 x
                gt_boxes3d[:,1,np.newaxis],   # 1 y
                gt_boxes3d[:,2,np.newaxis],   # 2 z
                gt_boxes3d[:,5,np.newaxis],   # 3 l # same as the original label
                gt_boxes3d[:,4,np.newaxis],   # 4 w # same as the original label
                gt_boxes3d[:,3,np.newaxis],   # 5 h # same as the original label
                gt_boxes3d[:,6,np.newaxis],   # 6 ry
            ), axis=1)
            
            if self.split=='train' or self.split =='test':
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(
                    gt_boxes3d, ['pedestrian' for i in range(len(gt_boxes3d))])

                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)

        if self.split=='train' or self.split =='test':
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            # return [pillars, voxels], [occupancy, position, size, angle, heading, classification] # network
            return [pillars, voxels], [occupancy, position, size, angle, heading] # network_v2
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
        if self.split=='train' or self.split =='test':
            self.sample_list=shuffle(self.sample_list)




class AnalyseCustomDataGenerator(CustomDataGenerator):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, batch_size: int, root_dir:str, point_cloud_statistics_path: str,
                npoints:int =16384, split: str ='train', 
                classes:List[str] =['Car', 'Pedestrian', 'Person_sitting'], random_select:bool =True,
                gt_database_dir=None, aug_hard_ratio:float=0.5, **kwargs):

        super(AnalyseCustomDataGenerator, self).__init__(
            batch_size=batch_size,  root_dir=root_dir, 
            point_cloud_statistics_path = point_cloud_statistics_path,
            npoints=npoints, split=split,   classes=classes, 
            random_select=random_select,    gt_database_dir=gt_database_dir, 
            aug_hard_ratio=aug_hard_ratio, **kwargs
        )
        self.batch_size = batch_size


    def get_sample(self, index):
        return super().get_sample(index)


    def __len__(self):
        return len(self.sample_list) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = range(batch_id * self.batch_size, self.batch_size * (batch_id + 1))
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



        for i in file_ids:
            point_cloud = self.get_lidar_without_background(i)

            pts_features = point_cloud[:, 3:]

            pts_input = np.concatenate([point_cloud[:,:3], pts_features[:,1,np.newaxis]], axis=1)

            # Voxels are the pillar ids
            pillars_, voxels_ = self.make_point_pillars(pts_input)

            pillars.append(pillars_)
            voxels.append(voxels_)

                

            
            if self.split=='train' or self.split =='test':
                obj_list = self.get_label(i) # are labels

                gt_boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
                # gt_bbox_params_list = []
                for k, obj in enumerate(obj_list):
                    gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                        = obj.pos, obj.h, obj.w, obj.l, limit_period(obj.ry, offset=0.5, period=2*np.pi) # mtr format
            #             = obj.pos, obj.h, obj.w, obj.l, obj.ry # kitti

            #     print(bboxes3d_[:,:3].shape)
                invalid_region_mask = self._get_invalid_region_mask(gt_boxes3d[:,:3])
                gt_boxes3d = gt_boxes3d[~invalid_region_mask,:]
                
                # for k in range(len(gt_boxes3d)):
                #     gt_bbox_params = [gt_boxes3d[k, 5], gt_boxes3d[k, 3], gt_boxes3d[k, 4],
                #                     gt_boxes3d[k, 1], gt_boxes3d[k, 2], gt_boxes3d[k, 0],
                #                     gt_boxes3d[k, 6]]

                #     gt_bbox_params_list.append(gt_bbox_params)

                
                
                
                # if gt_boxes3d.__len__() == 0:
                #     print('No gt object')
                #     continue

                gt_boxes3d = np.concatenate((
                    gt_boxes3d[:,0,np.newaxis],   # 0 x
                    gt_boxes3d[:,1,np.newaxis],   # 1 y
                    gt_boxes3d[:,2,np.newaxis],   # 2 z
                    gt_boxes3d[:,5,np.newaxis],   # 3 l # same as the original label
                    gt_boxes3d[:,4,np.newaxis],   # 4 w # same as the original label
                    gt_boxes3d[:,3,np.newaxis],   # 5 h # same as the original label
                    gt_boxes3d[:,6,np.newaxis],   # 6 ry
                ), axis=1)
                
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(
                    gt_boxes3d, ['pedestrian' for i in range(len(gt_boxes3d))])

                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

                gt_boxes3d_.append(gt_boxes3d)
                pts_input_.append(pts_input)
            elif self.split=='real':
                pts_input_.append(pts_input)


        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)

        if self.split=='train' or self.split =='test':
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            # return [pillars, voxels], [occupancy, position, size, angle, heading, classification] # network
            return [pillars, voxels], [occupancy, position, size, angle, heading], [pts_input_, gt_boxes3d_] # network_v2
        elif self.split=='real':
            return [pillars, voxels], [pts_input_]
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
        if self.split=='train' or self.split =='test':
            self.sample_list=shuffle(self.sample_list)