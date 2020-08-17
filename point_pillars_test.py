import unittest
import numpy as np
import tensorflow as tf

from point_pillars import createPillars, createPillarsTarget, select


class PointPillarsTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        xy = np.random.randint(-100, 100+1, size=(100000, 2))
        z = np.random.randint(-3, 1+1, size=(100000, 1))
        i = np.random.rand(100000)
        self.arr = np.c_[xy, z, i]
        assert self.arr.shape == (100000, 4)

    def test_pillar_creation(self):
        pillars, indices = createPillars(self.arr, 100, 12000, 0.16, 0.16, 0, 80.64, -40.32, 40.32, -3, 1, True)

        assert pillars.shape == (1, 12000, 100, 7)
        assert pillars.dtype == np.float32
        assert indices.shape == (1, 12000, 3)
        assert indices.dtype == np.int32

        session = tf.Session()
        pillars = tf.constant(pillars, dtype=tf.float32)
        indices = tf.constant(indices, dtype=tf.int32)
        feature_map = tf.scatter_nd(indices, tf.reduce_mean(pillars, axis=2), (1, 504, 504, 7))[0]
        arr, = session.run([feature_map])
        assert (arr.shape == (504, 504, 7))

    @staticmethod
    def test_pillar_target_creation():

        dims = np.array([[3.7, 1.6, 1.4], [3.7, 1.6, 1.4], [0.8, 0.6, 1.7]], dtype=np.float32)
        posn = np.array([[50, 10, 0], [20, 0, 0], [30, 5, 0]], dtype=np.float32)
        yaws = np.array([0, 0, 90], dtype=np.float32)

        target = createPillarsTarget(posn,
                                     dims,
                                     yaws,
                                     np.array([1, 1, 2], dtype=np.int32),
                                     dims[[0, 2]],
                                     np.array([0, 0], dtype=np.float32),
                                     np.array([0, 90], dtype=np.float32),
                                     0.5,
                                     0.4,
                                     10,
                                     2,
                                     0.1,
                                     0.1,
                                     0,
                                     80,
                                     -40,
                                     40,
                                     -3,
                                     1,
                                     True)

        assert target.shape == (3, 400, 400, 2, 10)
        assert (target[..., 0] == 1).sum() == 83

        selected = target[..., 0:1].argmax(axis=0)
        target = select(target, selected)
        assert (target.shape == (400, 400, 2, 10))


if __name__ == "__main__":
    unittest.main()
