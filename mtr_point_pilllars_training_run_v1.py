
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import numpy as np
import tensorflow as tf
from glob import glob

# from config import Parameters
from config_mtr_v1 import Parameters
from loss_v2_2 import PointPillarNetworkLoss
from network_v2_2 import build_point_pillar_graph
from mtr_processors_v1 import CustomDataGenerator
# from readers import KittiDataReader

from det3d.mtr_dataset import MTRDatasetBase

# from point_viz.converter import PointvizConverter

tf.get_logger().setLevel("ERROR")



# DATA_ROOT = "/media/data3/tjtanaa/kitti_dataset/KITTI/object/training"  # TODO make main arg
DATA_ROOT = "/media/data3/tjtanaa/Project4-MTR"  # TODO make main arg
MODEL_ROOT = "./logs_Pedestrian_MTR_No_Early_Stopping_wo_Aug_with_val"
PC_STATISTICS_PATH = "/home/tan/tjtanaa/det3d/det3d/mtr_dataset/point_cloud_statistics"

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# exit()
if __name__ == "__main__":
    params = Parameters()

    # gpus = tf.config.experimental.list_physical_devices('GPU')

    pillar_net = build_point_pillar_graph(params)
    # pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
    pillar_net.summary()
    # exit()
    loss = PointPillarNetworkLoss(params)

    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate, decay=params.decay_rate)

    pillar_net.compile(optimizer, loss=loss.losses())
    

    train_dataset = MTRDatasetBase(DATA_ROOT, 'train', PC_STATISTICS_PATH)

    gt_database_dir = os.path.join(DATA_ROOT, "gt_database")
    # gt_database_dir = None

    training_gen = CustomDataGenerator(batch_size=params.batch_size,root_dir = DATA_ROOT,
                    point_cloud_statistics_path=PC_STATISTICS_PATH,
                    npoints=8000, split='train',   classes=list(params.classes_map.keys()), 
                    random_select=True, gt_database_dir=gt_database_dir, aug_hard_ratio=0.7)

    validation_gen = CustomDataGenerator(batch_size=params.batch_size,  root_dir=DATA_ROOT, 
                    point_cloud_statistics_path=PC_STATISTICS_PATH,
                    npoints=8000, split='test', classes=list(params.classes_map.keys()))


    # save_viz_path = "/home/tan/tjtanaa/PointPillars/visualization/custom_processor"
    # Initialize and setup output directory.
    # Converter = PointvizConverter(save_viz_path)



    # bbox_params = self.convert_labels_into_point_viz_format(gt_boxes3d)
    # print(bbox_params.shape)
    # Converter.compile("custom_sample_{}".format(i), coors=pts_input[:,:3], intensity=pts_input[:,3],
    #                 bbox_params=bbox_params)
        

    log_dir = MODEL_ROOT
    epoch_to_decay = int(
        np.round(params.iters_to_decay / params.batch_size * int(len(training_gen))))
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, "model.h5"),
                                           monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.8 if ((epoch % epoch_to_decay == 0) and (epoch != 0)) else lr, verbose=True),
        # tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
    ]

    try:
        pillar_net.fit(training_gen,
                       validation_data = validation_gen,
                       steps_per_epoch=len(training_gen),
                       callbacks=callbacks,
                       use_multiprocessing=True,
                       max_queue_size = 16,
                       epochs=int(params.total_training_epochs),
                       workers=6)
    except KeyboardInterrupt:
        model_str = "interrupted_%s.h5" % time.strftime("%Y%m%d-%H%M%S")
        pillar_net.save(os.path.join(log_dir, model_str))
        print("Interrupt. Saving output to %s" % os.path.join(os.getcwd(), log_dir[1:], model_str))


