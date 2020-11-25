import numpy as np


class GridParameters:
    x_min = -10.08
    x_max = 10.08
    x_step = 0.04

    y_min = -10.08 #-5
    y_max = 10.08 #7.5
    y_step = 0.04

    # z_min = -1.0
    # z_max = 3.0
    z_min = -1.0
    z_max = 6.0

    # derived parameters
    Xn_f = float(x_max - x_min) / x_step
    Yn_f = float(y_max - y_min) / y_step
    Xn = int(Xn_f)
    Yn = int(Yn_f)

    def __init__(self, **kwargs):
        super(GridParameters, self).__init__(**kwargs)


class DataParameters:

    # classes_map = {"Car":               0,
    #            "Pedestrian":        1,
    #            "Person_sitting":    1,
    #            "Cyclist":           2,
    #            "Truck":             3,
    #            "Van":               3,
    #            "Tram":              3,
    #            "Misc":              3,
    #            }

    # for Car and Pedestrian
    # map_classes = {
    #     0: "Car",
    #     1: "Pedestrian"
    # }

    # classes_map = {"Car":               0,
    #            "Pedestrian":        1,
    #            "Person_sitting":    1,
    #         #    "Cyclist":           2,
    #         #    "Truck":             3,
    #         #    "Van":               3,
    #         #    "Tram":              3,
    #         #    "Misc":              3,
    #            }


    # for Car only
    map_classes = {
        0: "pedestrian"
    }

    classes_map = {"pedestrian":               0
               }

    # # for Pedestrian only
    # map_classes = {
    #     0: "Pedestrian"
    # }

    # classes_map = {
    #            "Pedestrian":        0,
    #            "Person_sitting":    0,
    #            }

    nb_classes = len(np.unique(list(classes_map.values())))
    assert nb_classes == np.max(np.unique(list(classes_map.values()))) + 1, 'Starting class indexing at zero.'

    # classes = {"Car":               0,
    #            "Pedestrian":        1,
    #            "Person_sitting":    1,
    #            "Cyclist":           2,
    #            "Truck":             3,
    #            "Van":               3,
    #            "Tram":              3,
    #            "Misc":              3,
    #            }

    # nb_classes = len(np.unique(list(classes.values())))
    # assert nb_classes == np.max(np.unique(list(classes.values()))) + 1, 'Starting class indexing at zero.'

    def __init__(self, **kwargs):
        super(DataParameters, self).__init__(**kwargs)


class NetworkParameters:

    max_points_per_pillar = 100
    max_pillars = 12000
    nb_features = 9
    nb_channels = 64
    downscaling_factor = 2

    # length (x), width (y), height (z), z-center, orientation
    # for car and pedestrian
    # anchor_dims = np.array([[3.9, 1.6, 1.56, -1, 0],
    #                         [3.9, 1.6, 1.56, -1, np.pi/2],
    #                         [0.8, 0.6, 1.73, -0.6, 0],
    #                         [0.8, 0.6, 1.73, -0.6, np.pi/2],
    #                         ], dtype=np.float32).tolist()

    # for car only
    # anchor_dims = np.array([[3.9, 1.6, 1.56, -1, 0],
    #                         [3.9, 1.6, 1.56, -1, np.pi/2]], dtype=np.float32).tolist()

    # for pedestrian only
    anchor_dims = np.array([
                            [0.62, 0.56, 0.7, 1.8, 0],
                            [0.62, 0.56, 0.7, 1.8, np.pi/2],
                            [0.62, 0.56, 1.5, 1.63646424, 0],
                            [0.62, 0.56, 1.5, 1.63646424, np.pi/2],
                            ], dtype=np.float32).tolist()
    nb_dims = 3
    
    # for car
    # positive_iou_threshold = 0.6
    # negative_iou_threshold = 0.3
    
    # for pedestrian
    positive_iou_threshold = 0.5
    negative_iou_threshold = 0.35

    # batch_size = 1
    num_gpus = 1
    batch_size = 4 
    total_training_epochs = 160
    # iters_to_decay = 101040.    # 15 * 4 * ceil(6733. / 4) --> every 15 epochs on 6733 kitti samples, cf. pillar paper 
    iters_to_decay = 100500
    learning_rate = 2e-4
    decay_rate = 1e-8
    L1 = 0
    L2 = 0
    alpha = 0.25
    gamma = 2.0
                            # original pillars paper values
    focal_weight = 1.0      # 1.0
    loc_weight = 2.0        # 2.0
    size_weight = 2.0       # 2.0
    angle_weight = 2.0      # 2.0
    heading_weight = 0.2    # 0.2
    class_weight = 0.5      # 0.2

    def __init__(self, **kwargs):
        super(NetworkParameters, self).__init__(**kwargs)


class Parameters(GridParameters, DataParameters, NetworkParameters):

    def __init__(self, **kwargs):
        super(Parameters, self).__init__(**kwargs)
