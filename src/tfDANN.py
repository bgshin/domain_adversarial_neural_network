import tensorflow as tf
import numpy as np

from flip_gradient import flip_gradient

class DANN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,sequence_length, num_classes, embedding_size, batch_size):
        self.input_x = tf.placeholder(tf.int32, [None, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.adapt_lambda = tf.placeholder(tf.float32, [])
        print(self.input_x.get_shape())

        self.X = tf.cast(self.input_x, tf.float32)
        with tf.variable_scope('feature_extractor'):
            W_0 = tf.Variable(tf.truncated_normal([embedding_size, 200], stddev=0.1))
            b_0 = tf.constant(0.1, shape=[200])
            h_0 = tf.nn.relu(tf.matmul(self.X, W_0) + b_0)

            W_1 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
            b_1 = tf.constant(0.1, shape=[200])
            h_1 = tf.nn.relu(tf.matmul(h_0, W_1) + b_1)

            self.feature = h_1
            print(self.feature.get_shape())

        with tf.variable_scope('label_predictor'):
            lp_V_0 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
            lp_c_0 = tf.constant(0.1, shape=[200])
            lp_h_0 = tf.nn.relu(tf.matmul(self.feature, lp_V_0) + lp_c_0)

            lp_V_1 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
            lp_c_1 = tf.constant(0.1, shape=[200])
            lp_h_1 = tf.nn.relu(tf.matmul(lp_h_0, lp_V_1) + lp_c_1)

            lp_V_2 = tf.Variable(tf.truncated_normal([200, 2], stddev=0.1))
            lp_c_2 = tf.constant(0.1, shape=[2])
            lp_logits = tf.matmul(lp_h_1, lp_V_2) + lp_c_2

            label_predictions = tf.nn.softmax(lp_logits)
            print(label_predictions.get_shape())

        with tf.name_scope("label_predictor_loss"):
            label_pred_losses = tf.nn.softmax_cross_entropy_with_logits(label_predictions, self.input_y)
            self.mean_label_pred_loss = tf.reduce_mean(label_pred_losses)

        with tf.name_scope("label_predictor_accuracy"):
            lp_correct_predictions = tf.equal(tf.argmax(self.input_y, 1), tf.argmax(label_predictions , 1))
            self.label_pred_accuracy = tf.reduce_mean(tf.cast(lp_correct_predictions, "float"), name="label_predictor_accuracy")


        with tf.variable_scope("GRL"):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.adapt_lambda) #TODO: Change to own implementation

            #TODO: Generalize code. All values are hardcoded below - shapes, hidden layer, final layer,

        with tf.variable_scope("domain_predictor"):

            dp_U_0 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
            dp_d_0 = tf.constant(0.1, shape=[100])
            dp_h_0 = tf.nn.relu(tf.matmul(feat, dp_U_0) + dp_d_0)

            dp_U_1 = tf.Variable(tf.truncated_normal([100, 2], stddev=0.1))
            dp_d_1 = tf.constant(0.1, shape=[2])
            dp_logits = tf.matmul(dp_h_0, dp_U_1) + dp_d_1

            domain_predictions = tf.nn.softmax(dp_logits)
            print(domain_predictions.get_shape())
        with tf.name_scope("domain_predictor_loss"):
            domain_pred_losses = tf.nn.softmax_cross_entropy_with_logits(domain_predictions, self.domain)
            self.mean_domain_pred_loss = tf.reduce_mean(domain_pred_losses) #+ l2_reg_lambda * l2_loss

        with tf.name_scope("domain_predictor_accuracy"):
            dp_correct_predictions = tf.equal(tf.argmax(self.domain, 1), tf.argmax(domain_predictions , 1))
            self.domain_pred_accuracy = tf.reduce_mean(tf.cast(dp_correct_predictions, "float"), name="domain_predictor_accuracy")


        with tf.name_scope("total_loss"):
            self.total_loss = self.mean_label_pred_loss + self.mean_domain_pred_loss