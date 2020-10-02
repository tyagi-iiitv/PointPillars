import tensorflow as tf
from tensorflow.python.keras import backend as K
from config import Parameters
import sys


class BoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, params: Parameters):
        super(BoxLoss, self).__init__(
            reduction="none", name="BoxLoss"
        )
        self.delta= float(params.delta)

    def call(self, y_true, y_pred):
        print('box_y_true',y_true)
        print('box_y_pred',y_pred)
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self.delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class ClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, params: Parameters):
        super(ClassificationLoss, self).__init__(
            reduction="none", name="ClassificationLoss"
        )        
        self.alpha = float(params.alpha)
        self.gamma = float(params.gamma)

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self.alpha, (1.0 - self.alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class PointPillarNetworkLoss:

    def __init__(self, params: Parameters):
        self.alpha = float(params.alpha)
        self.gamma = float(params.gamma)
        self._delta =float(params.delta)
        self.focal_weight = float(params.focal_weight)
        self.loc_weight = float(params.loc_weight)
        self.size_weight = float(params.size_weight)
        self.angle_weight = float(params.angle_weight)
        self.heading_weight = float(params.heading_weight)
        self.class_weight = float(params.class_weight)
        self._num_classes = int(params.nb_classes)
        self._clf_loss = ClassificationLoss(params)
        self._box_loss = BoxLoss(params)

    def losses(self):#,y_true: tf.Tensor, y_pred: tf.Tensor):
        #print('y_true____________________',y_true)
        #print('y_pred____________________',y_pred)
        return [self.total_loss]

    
    def total_loss(self,y_true: tf.Tensor, y_pred: tf.Tensor):
        #print('y_pred_______________________',y_pred[:0])
        #print('y_pred_______________________',y_pred[:,:1])
        #print('y_pred_______________________',y_pred[2])
        #print('y_pred_______________________',y_pred[3])
        #print('y_pred_______________________',y_pred[4])
        #print(y_true)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
        box_labels = y_true[...,1:4]
        print('BOX_LABELS____________________',box_labels)
        box_predictions = y_pred[...,1:4]
        print('box_predictions_______________',box_predictions)
        
        cls_predictions = y_pred[...,8:13]        
        clf_loss = self._clf_loss(y_true[...,8:13], cls_predictions)
        mask=tf.cast(tf.greater(y_true[...,0], -1.0), dtype=tf.float32)
        #positive_mask = tf.cast(tf.greater(y_true[...,0], -1.0), dtype=tf.float32)
        #ignore_mask = tf.cast(tf.equal(y_true[...,1], -2.0), dtype=tf.float32)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(mask, 0.0), 0.0, clf_loss)
        clf_loss = tf.where(tf.equal(mask, -1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss

    
