import os
import time
import numpy as np
import tensorflow as tf
from glob import glob

from config import Parameters
from loss import PointPillarNetworkLoss
from network import build_point_pillar_graph
from processors import SimpleDataGenerator
from readers import KittiDataReader

tf.get_logger().setLevel("ERROR")

DATA_ROOT = "../training"  # TODO make main arg
MODEL_ROOT = "./logs"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == "__main__":

    params = Parameters()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    loss = PointPillarNetworkLoss(params)
    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate, decay=params.decay_rate)
    if len(gpus)>1:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        with strategy.scope():
            pillar_net = build_point_pillar_graph(params)
            pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
            pillar_net.compile(optimizer, loss=loss.losses())
    else:
        pillar_net = build_point_pillar_graph(params)
        pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
        pillar_net.compile(optimizer, loss=loss.losses())

    data_reader = KittiDataReader()

    lidar_files = sorted(glob(os.path.join(DATA_ROOT, "velodyne", "*.bin")))
    label_files = sorted(glob(os.path.join(DATA_ROOT, "label_2", "*.txt")))
    calibration_files = sorted(glob(os.path.join(DATA_ROOT, "calib", "*.txt")))
    assert len(lidar_files) == len(label_files) == len(calibration_files), "Input dirs require equal number of files."
    validation_len = int(0.3*len(label_files))
    
    training_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files[:-validation_len], label_files[:-validation_len], calibration_files[:-validation_len])
    validation_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files[-validation_len:], label_files[-validation_len:], calibration_files[-validation_len:])

    log_dir = MODEL_ROOT
    epoch_to_decay = int(
        np.round(params.iters_to_decay / params.batch_size * int(np.ceil(float(len(label_files)) / params.batch_size))))
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, "model.h5"),
                                           monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.8 if ((epoch % epoch_to_decay == 0) and (epoch != 0)) else lr, verbose=True),
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
    ]

    try:
        pillar_net.fit(training_gen,
                       validation_data = validation_gen,
                       steps_per_epoch=len(training_gen),
                       callbacks=callbacks,
                       use_multiprocessing=True,
                       epochs=int(params.total_training_epochs),
                       workers=6)
    except KeyboardInterrupt:
        model_str = "interrupted_%s.h5" % time.strftime("%Y%m%d-%H%M%S")
        pillar_net.save(os.path.join(log_dir, model_str))
        print("Interrupt. Saving output to %s" % os.path.join(os.getcwd(), log_dir[1:], model_str))
