import tensorflow as tf
import numpy as np

from flip_gradient import flip_gradient

class DANN(object):
    def __init__(self, num_classes, embedding_size, hidden_layer_size):
        self.xs = tf.placeholder(tf.int32, [None, embedding_size], name="x_source")
        self.ys = tf.placeholder(tf.float32, [None, num_classes], name="y_source")
        self.xt = tf.placeholder(tf.float32, [None, embedding_size], name="x_target")
        self.domain_s = tf.placeholder(tf.float32, [None, 2], name="domain_label_source")
        self.domain_t = tf.placeholder(tf.float32, [None, 2], name="domain_label_target")
        self.adapt_lambda = tf.placeholder(tf.float32, [])

        self.xs = tf.cast(self.xs, tf.float32)
        self.xt = tf.cast(self.xt, tf.float32)

        print 'self.xs.get_shape()', self.xs.get_shape()
        print 'self.xt.get_shape()', self.xt.get_shape()
        print 'self.domain_s.get_shape()', self.domain_s.get_shape()
        print 'self.domain_t.get_shape()', self.domain_t.get_shape()

        with tf.variable_scope('vars'):
            W = tf.Variable(tf.truncated_normal([embedding_size, hidden_layer_size], stddev=0.1), name='W')
            b = tf.Variable(tf.truncated_normal([hidden_layer_size], stddev=0.1), name='b')
            # b = tf.constant(0.1, shape=[hidden_layer_size])
            V = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], stddev=0.1), name='V')
            c = tf.Variable(tf.truncated_normal([num_classes], stddev=0.1), name='c')
            # c = tf.constant(0.1, shape=[num_classes])
            U = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], stddev=0.1), name='U')
            d = tf.Variable(tf.truncated_normal([num_classes], stddev=0.1), name='d')
            # d = tf.constant(0.1, shape=[num_classes])

        # with tf.variable_scope('Feature_Extractor'):
        #     W = tf.Variable(tf.truncated_normal([embedding_size, hidden_layer_size], stddev=0.1))
        #     b = tf.constant(0.1, shape=[hidden_layer_size])
        #
        # with tf.variable_scope('Classifier'):
        #     V = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], stddev=0.1))
        #     c = tf.constant(0.1, shape=[num_classes])
        #
        # with tf.variable_scope('Domain_Discriminator'):
        #     U = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], stddev=0.1))
        #     d = tf.constant(0.1, shape=[num_classes])

        # with tf.variable_scope('Classifier_Path'):
        # with tf.variable_scope('Feature_Extractor', reuse=True), tf.variable_scope('Classifier', reuse=True):
        # W = tf.get_variable("W", [embedding_size, hidden_layer_size])
        # b = tf.get_variable("b", [ hidden_layer_size])
        # V = tf.get_variable("V", [hidden_layer_size, num_classes])
        # c = tf.get_variable("c", [num_classes])

        self.feature_xs = tf.nn.sigmoid(tf.matmul(self.xs, W) + b)
        print 'self.feature_xs.get_shape()', self.feature_xs.get_shape()
        self.GyGfXs = tf.nn.softmax(tf.matmul(self.feature_xs, V) + c)
        print 'self.GyGfXs.get_shape()', self.GyGfXs.get_shape()


        # with tf.variable_scope('Domain_Discriminator_Path'):
        # with tf.variable_scope('Feature_Extractor', reuse=True), tf.variable_scope('Domain_Discriminator', reuse=True):
        # W = tf.get_variable("W", [embedding_size, hidden_layer_size])
        # b = tf.get_variable("b", [hidden_layer_size])
        # U = tf.get_variable("U", [hidden_layer_size, num_classes])
        # d = tf.get_variable("d", [num_classes])

        self.GdGfXs = tf.nn.softmax(tf.matmul(self.feature_xs, U) + d)
        self.GdGfXs_sigmoid = tf.nn.sigmoid(tf.matmul(self.feature_xs, U) + d)
        print 'self.GdGfXs.get_shape()', self.GdGfXs.get_shape()

        self.feature_xt = tf.nn.sigmoid(tf.matmul(self.xt, W) + b)
        print 'self.feature_xt.get_shape()', self.feature_xt.get_shape()
        self.GdGfXt = tf.nn.softmax(tf.matmul(self.feature_xt, U) + d)
        self.GdGfXt_sigmoid = tf.nn.sigmoid(tf.matmul(self.feature_xt, U) + d)
        print 'self.GdGfXt.get_shape()', self.GdGfXt.get_shape()

        self.classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.GyGfXs, self.ys))

        # self.discriminator_source_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.GdGfXs, self.domain_s))
        # self.discriminator_target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.GdGfXt, self.domain_t))

        self.discriminator_source_loss = tf.reduce_mean(tf.log(self.GdGfXs_sigmoid))
        self.discriminator_target_loss = tf.reduce_mean(tf.log(1-self.GdGfXt_sigmoid))



        self.total_cost = self.classifier_loss - \
                          self.adapt_lambda* (self.discriminator_source_loss+self.discriminator_target_loss)

        self.total_neg_cost = -self.total_cost

        self.classifier_predictions = tf.equal(tf.argmax(self.ys, 1), tf.argmax(self.GyGfXs, 1))
        self.classifier_acc = tf.reduce_mean(tf.cast(self.classifier_predictions, "float"),
                                                   name="classifier_acc")


        # self.discriminator_predictions_s = tf.equal(tf.argmax(self.domain_s, 1), tf.argmax(self.GdGfXs, 1))
        # self.discriminator_accuracy_s = tf.reduce_mean(tf.cast(self.discriminator_predictions_s, "float"),
        #                                                name="discriminator_accuracy_s")

        self.discriminator_predictions_s = tf.greater_equal(self.GdGfXs, 0.5)
        self.discriminator_accuracy_s = tf.reduce_mean(tf.cast(self.discriminator_predictions_s, "float"),
                                                   name="discriminator_accuracy_s")

        print 'self.domain_t', self.domain_t
        print 'self.GdGfXt', self.GdGfXt

        # self.discriminator_predictions_t = tf.equal(tf.argmax(self.domain_t, 1), tf.argmax(self.GdGfXt, 1))
        # self.discriminator_accuracy_t = tf.reduce_mean(tf.cast(self.discriminator_predictions_t, "float"),
        #                                                name="discriminator_accuracy_t")

        self.discriminator_predictions_t = tf.greater_equal(self.GdGfXs, 0.5)
        self.discriminator_accuracy_t = tf.reduce_mean(tf.cast(self.discriminator_predictions_t, "float"),
                                                       name="discriminator_accuracy_t")

        self.discriminator_acc = (self.discriminator_accuracy_s + self.discriminator_accuracy_t)/2

        print 'self.classifier_loss', self.classifier_loss
        print 'self.discriminator_source_loss', self.discriminator_source_loss
        print 'self.discriminator_target_loss', self.discriminator_target_loss
        print 'self.total_cost', self.total_cost
        print 'self.total_neg_cost', self.total_neg_cost

        print 'self.classifier_acc', self.classifier_acc
        print 'self.discriminator_acc', self.discriminator_acc


class DANNold(object):
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