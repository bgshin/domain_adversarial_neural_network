import numpy as np

from sklearn.datasets import load_svmlight_files
from sklearn import svm
import tensorflow as tf
import numpy as np
import os
import time
import datetime
#import data_helpers
from tfDANN import DANN
from tensorflow.contrib import learn
from utils import batch_generator


def load_amazon(source_name, target_name, data_folder=None):
    """
    Load the amazon sentiment datasets from svmlight format files
    inputs:
        source_name : name of the source dataset
        target_name : name of the target dataset
        data_folder : path to the folder containing the files
    outputs:
        xs : training source data matrix
        ys : training source label vector
        xt : training target data matrix
        yt : training target label vector
        xtest : testing target data matrix
        ytest : testing target label vector
    """

    if data_folder is None:
        data_folder = '../data/'

    source_file = data_folder + source_name + '_train.svmlight'
    target_file = data_folder + target_name + '_train.svmlight'
    test_file = data_folder + target_name + '_test.svmlight'

    xs, ys, xt, yt, xtest, ytest = load_svmlight_files([source_file, target_file, test_file])

    # Convert sparse matrices to numpy 2D array
    xs, xt, xtest = (np.array(X.todense()) for X in  (xs, xt, xtest))

    # Convert {-1,1} labels to {0,1} labels
    ys, yt, ytest = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, ytest))



    return xs, ys, xt, yt, xtest, ytest

# def compute_proxy_distance(source_X, target_X, verbose=False):
#     """
#     Compute the Proxy-A-Distance of a source/target representation
#     """
#     nb_source = np.shape(source_X)[0]
#     nb_target = np.shape(target_X)[0]
#
#     if verbose:
#         print('PAD on', (nb_source, nb_target), 'examples')
#
#     C_list = np.logspace(-5, 4, 10)
#
#     half_source, half_target = int(nb_source/2), int(nb_target/2)
#     train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
#     train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))
#
#     test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
#     test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))
#
#     best_risk = 1.0
#     for C in C_list:
#         clf = svm.SVC(C=C, kernel='linear', verbose=False)
#         clf.fit(train_X, train_Y)
#
#         train_risk = np.mean(clf.predict(train_X) != train_Y)
#         test_risk = np.mean(clf.predict(test_X) != test_Y)
#
#         if verbose:
#             print('--------- [ C =', C, '] ---------')
#             print('PAD train risk: ', train_risk)
#             print('PAD test risk:  ', test_risk)
#
#         if test_risk > .5:
#             test_risk = 1. - test_risk
#
#         best_risk = min(best_risk, test_risk)
#
#     return 2 * (1. - 2 * best_risk)

def train(xs_train, ys_train, xt_train, yt_train, xs_dev, ys_dev, xt_dev, yt_dev):

    #Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 5000, "Dimensionality of character embedding (default: 5000")
    tf.flags.DEFINE_float("adapt_lambda", 1.0, "Domain classifier regularizaion lambda (default: 1.0)")
    tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate (default: 0.05)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    graph = tf.get_default_graph()
    with graph.as_default():
        model = DANN(
                num_classes= ys_train.shape[1],
                embedding_size=FLAGS.embedding_dim,
                hidden_layer_size=50)

        global_step = tf.Variable(0, name="global_step", trainable=False)

        # train_a = tf.train.GradientDescentOptimizer(0.1).minimize(loss_a, var_list=[A])
        # train_b = tf.train.GradientDescentOptimizer(0.1).minimize(loss_b, var_list=[B])

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vars")
        var_list_classifier = train_vars[0:4]
        var_list_discriminator = train_vars[4:]

        # op_classifier = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.total_cost,
        #                                                                      var_list=var_list_classifier)
        # op_discriminator = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.total_neg_cost,
        #                                                                      var_list=var_list_discriminator)

        op_classifier = tf.train.AdamOptimizer(FLAGS.learning_rate)
        op_discriminator = tf.train.AdamOptimizer(FLAGS.learning_rate)

        grad_classifier = op_classifier.compute_gradients(model.total_cost, var_list=var_list_classifier)
        grad_discriminator = op_discriminator.compute_gradients(model.total_neg_cost, var_list=var_list_discriminator)

        train_op_classifier = op_classifier.apply_gradients(grad_classifier, global_step=global_step)
        train_op_discriminator = op_discriminator.apply_gradients(grad_discriminator, global_step=global_step)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.Saver(tf.global_variables())


        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()

            def train_step(xs_batch, ys_batch, xt_batch, ds_batch, dt_batch, lam):
                """
                A single training step
                """
                feed_dict = {
                    model.xs: xs_batch,
                    model.ys: ys_batch,
                    model.xt: xt_batch,
                    model.domain_s: ds_batch,
                    model.domain_t: dt_batch,
                    model.adapt_lambda: lam,
                }
                _, _, step, classifier_loss, discriminator_source_loss, discriminator_target_loss, total_cost, \
                classifier_acc, discriminator_acc = sess.run(
                    [train_op_classifier, train_op_discriminator, global_step, model.classifier_loss,
                     model.discriminator_source_loss, model.discriminator_target_loss, model.total_cost,
                     model.classifier_acc, model.discriminator_acc], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(
                "{}: step {}, c_loss {:g}, ds_loss {:g}, dt_loss {:g}, loss{:g}, c_acc {:g}, d_acc {:g}".format(
                    time_str, step, classifier_loss, discriminator_source_loss, discriminator_target_loss, total_cost,
                    classifier_acc, discriminator_acc))

            def dev_step(xs_batch, ys_batch, xt_batch, ds_batch, dt_batch, lam):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    model.xs: xs_batch,
                    model.ys: ys_batch,
                    model.xt: xt_batch,
                    model.domain_s: ds_batch,
                    model.domain_t: dt_batch,
                    model.adapt_lambda: lam,
                }
                step, classifier_loss, discriminator_source_loss, discriminator_target_loss, total_cost, \
                classifier_acc, discriminator_acc = sess.run(
                    [global_step, model.classifier_loss, model.discriminator_source_loss,
                     model.discriminator_target_loss, model.total_cost,
                     model.classifier_acc, model.discriminator_acc], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(
                    "{}: step {}, c_loss {:g}, ds_loss {:g}, dt_loss {:g}, loss{:g}, c_acc {:g}, d_acc {:g}".format(
                        time_str, step, classifier_loss, discriminator_source_loss, discriminator_target_loss,
                        total_cost,
                        classifier_acc, discriminator_acc))

            # tf.initialize_all_variables().run()


            gen_source_batch = batch_generator([xs_train, ys_train], FLAGS.batch_size / 2)
            gen_target_batch = batch_generator([xt_train, yt_train], FLAGS.batch_size / 2)

            domain_labels_s = np.tile([1., 0.], [FLAGS.batch_size / 2, 1])
            domain_labels_t = np.tile([0., 1.], [FLAGS.batch_size / 2, 1])
            # domain_labels = np.vstack(
            #     [np.tile([1., 0.], [FLAGS.batch_size / 2, 1]), np.tile([0., 1.], [FLAGS.batch_size / 2, 1])])

            if ((len(xs_train) + len(xt_train)) % FLAGS.batch_size > 0):
                num_batches = ((len(xs_train) + len(xt_train)) / FLAGS.batch_size) + 1
            else:
                num_batches = ((len(xs_train) + len(xt_train)) / FLAGS.batch_size)

            for epoch in range(FLAGS.num_epochs):
                for i in range(num_batches):
                    p = float(i) / num_batches
                    l = 2. / (1. + np.exp(-10. * p)) - 1
                    lr = 0.01 / (1. + 10 * p) ** 0.75

                    Xs, Ys = gen_source_batch.next()
                    Xt, Yt = gen_target_batch.next()

                    train_step(Xs, Ys, Xt, domain_labels_s, domain_labels_t, l)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        ds_label = np.tile([1., 0.], [len(ys_dev), 1])
                        dt_label = np.tile([1., 0.], [len(yt_dev), 1])

                        print("\nEvaluation:")
                        dev_step(xs_dev, ys_dev, xt_dev, ds_label, dt_label, l)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))


def test(xtest, ytest_hot):
    print 'd'

def main():
    data_folder = '../data/' # where the datasets are
    source_name = 'books'   # source domain: books, dvd, kitchen, or electronics
    target_name = 'dvd'     # traget domain: books, dvd, kitchen, or electronics


    ##Loading pre-embedded data
    print("Loading data ...")
    xs, ys, xt, yt, xtest, ytest = load_amazon(source_name, target_name, data_folder)
    print xs.shape
    print xt.shape

    #Define validation sets
    nb_valid = int(0.1 * len(ys))
    nb_valid_s = nb_valid / 2
    nb_valid_t = nb_valid / 2

    # x_valid, y_valid = xs[-nb_valid:, :], ys[-nb_valid:]

    ys_hot = np.zeros((len(ys), 2))
    ys_hot[np.arange(len(ys)), ys] = 1
    yt_hot = np.zeros((len(yt), 2))
    yt_hot[np.arange(len(yt)), yt] = 1

    x_valid_s, y_valid_s_hot = xs[-nb_valid_s:, :], ys_hot[-nb_valid_s:, :]
    x_valid_t, y_valid_t_hot = xt[-nb_valid_s:, :], yt_hot[-nb_valid_s:, :]

    xs, ys = xs[0:-nb_valid_s, :], ys[0:-nb_valid_s]
    xt, yt = xt[0:-nb_valid_s, :], yt[0:-nb_valid_s]
    print xs.shape
    print xt.shape

    ytest_hot = np.zeros((len(ytest), 2))
    ytest_hot[np.arange(len(ytest)), ytest] = 1


    print("Training...")
    train(xs, ys_hot, xt, yt_hot, x_valid_s, y_valid_s_hot, x_valid_t, y_valid_t_hot)

    print("Evaluating...")
    # test(xtest, ytest_hot)


    #
    # print('Training Risk   = %f' % np.mean(prediction_train != ys))
    # print('Validation Risk = %f' % np.mean(prediction_valid != yv))
    # print('Test Risk       = %f' % np.mean(prediction_test != ytest))

    # print('==================================================================')
    #
    # print('Computing PAD on original data...')
    # pad_original = compute_proxy_distance(xs, xt, verbose=True)
    #
    # print('Computing PAD on DANN representation...')
    # pad_dann = compute_proxy_distance(algo.hidden_representation(xs), algo.hidden_representation(xt), verbose=True)
    #
    # print('PAD on original data       = %f' % pad_original)
    # print('PAD on DANN representation = %f' % pad_dann)



if __name__ == '__main__':
    main()
