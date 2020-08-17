import tensorflow as tf
from tensorflow.python.keras import backend as K
from config import Parameters


class PointPillarNetworkLoss:

    def __init__(self, params: Parameters):
        self.alpha = float(params.alpha)
        self.gamma = float(params.gamma)
        self.focal_weight = float(params.focal_weight)
        self.loc_weight = float(params.loc_weight)
        self.size_weight = float(params.size_weight)
        self.angle_weight = float(params.angle_weight)
        self.heading_weight = float(params.heading_weight)
        self.class_weight = float(params.class_weight)

    def losses(self):
        return [self.focal_loss, self.loc_loss, self.size_loss, self.angle_loss, self.heading_loss, self.class_loss]

    def focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):

        self.mask = tf.equal(y_true, 1)

        cross_entropy = K.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + (tf.subtract(1.0, y_true) * tf.subtract(1.0, y_pred))

        gamma_factor = tf.pow(1.0 - p_t, self.gamma)

        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        focal_loss = gamma_factor * alpha_factor * cross_entropy

        mask = tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 1))
        masked_loss = tf.boolean_mask(focal_loss, mask)

        return self.focal_weight * tf.reduce_mean(masked_loss)

    def loc_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        mask = tf.tile(tf.expand_dims(self.mask, -1), [1, 1, 1, 1, 3])
        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")

        masked_loss = tf.boolean_mask(loss, mask)
        return self.loc_weight * tf.reduce_mean(masked_loss)

    def size_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        mask = tf.tile(tf.expand_dims(self.mask, -1), [1, 1, 1, 1, 3])
        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")

        masked_loss = tf.boolean_mask(loss, mask)
        return self.size_weight * tf.reduce_mean(masked_loss)

    def angle_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")

        masked_loss = tf.boolean_mask(loss, self.mask)
        return self.angle_weight * tf.reduce_mean(masked_loss)

    def heading_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = K.binary_crossentropy(y_true, y_pred)
        masked_loss = tf.boolean_mask(loss, self.mask)
        return self.heading_weight * tf.reduce_mean(masked_loss)

    def class_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred, reduction="none")
        masked_loss = tf.boolean_mask(loss, self.mask)
        return self.class_weight * tf.reduce_mean(masked_loss)
