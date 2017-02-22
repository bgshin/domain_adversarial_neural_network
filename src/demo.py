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

def train(xs_train, ys_train, xt_train, yt_train, x_dev, y_dev):

    #Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 5000, "Dimensionality of character embedding (default: 5000")
    tf.flags.DEFINE_float("adapt_lambda", 1.0, "Domain classifier regularizaion lambda (default: 1.0)")
    tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate (default: 0.05)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
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
                sequence_length=xs_train.shape[1],
                num_classes= ys_train.shape[1],
                embedding_size=FLAGS.embedding_dim,
                batch_size=FLAGS.batch_size)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Summaries for loss and accuracy
        label_pred_loss_summary = tf.scalar_summary("mean_label_pred_loss", model.mean_label_pred_loss)
        label_pred_acc_summary = tf.scalar_summary("label_pred_accuracy", model.label_pred_accuracy)

        domain_pred_loss_summary = tf.scalar_summary("mean_domain_pred_loss", model.mean_domain_pred_loss)
        domain_pred_acc_summary = tf.scalar_summary("domain_pred_accuracy", model.domain_pred_accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([label_pred_loss_summary,
                                             domain_pred_loss_summary,
                                             label_pred_acc_summary,
                                             domain_pred_acc_summary,
                                             grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")

        # Dev summaries
        dev_summary_op = tf.merge_summary([label_pred_loss_summary,
                                           domain_pred_loss_summary,
                                           label_pred_acc_summary,
                                           domain_pred_acc_summary,
                                           ])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())



        with tf.Session(graph=graph) as sess:
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, graph)
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, graph)

            def train_step(x_batch, y_batch, batch_domain_labels, l, lr):
                """
                A single training step
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.domain: batch_domain_labels,
                    model.adapt_lambda: l,
                }
                _, step, summaries, batch_loss, ploss, dloss, p_acc, d_acc = sess.run(
                    [train_op, global_step, train_summary_op, model.total_loss, model.mean_label_pred_loss,
                     model.mean_domain_pred_loss, model.label_pred_accuracy, model.domain_pred_accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(
                "{}: step {}, batch_loss {:g}, dloss {:g}, ploss {:g}, dp_acc {:g}, lp_acc {:g}".format(time_str, step,
                                                                                                      batch_loss, dloss,
                                                                                                      ploss, d_acc, p_acc))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, batch_domain_labels, l, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.domain: batch_domain_labels,
                    model.adapt_lambda: l,

                    # model.dropout_keep_prob: 1.0
                }
                step, summaries, batch_loss, dloss, ploss, p_acc, d_acc = sess.run(
                    [global_step, dev_summary_op, model.total_loss, model.mean_label_pred_loss,
                     model.mean_domain_pred_loss, model.label_pred_accuracy, model.domain_pred_accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(
                "{}: step {}, batch_loss {:g}, dloss {:g}, ploss {:g}, d_acc {:g}, p_acc {:g}".format(time_str, step,
                                                                                                      batch_loss, dloss,
                                                                                                      ploss, d_acc,
                                                                                                      p_acc))
                if writer:
                    writer.add_summary(summaries, step)


            tf.initialize_all_variables().run()

            gen_source_batch = batch_generator([xs_train, ys_train], FLAGS.batch_size / 2)
            gen_target_batch = batch_generator([xt_train, yt_train], FLAGS.batch_size / 2)

            domain_labels = np.vstack(
                [np.tile([1., 0.], [FLAGS.batch_size / 2, 1]), np.tile([0., 1.], [FLAGS.batch_size / 2, 1])])

            if ((len(xs_train) + len(xt_train)) % FLAGS.batch_size > 0):
                num_batches = ((len(xs_train) + len(xt_train)) / FLAGS.batch_size) + 1
            else:
                num_batches = ((len(xs_train) + len(xt_train)) / FLAGS.batch_size)

            for i in range(num_batches):
                p = float(i) / num_batches
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p) ** 0.75

                X0, y0 = gen_source_batch.next()
                X1, y1 = gen_target_batch.next()
                X = np.vstack([X0, X1])
                Y = np.vstack([y0, y1])

                train_step(X, Y, domain_labels, l, lr)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    d_labels = np.vstack(
                [np.tile([1., 0.], [len(y_dev) / 2, 1]), np.tile([0., 1.], [len(x_dev) / 2, 1])])
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, d_labels, l, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main():
    data_folder = './data/' # where the datasets are
    source_name = 'books'   # source domain: books, dvd, kitchen, or electronics
    target_name = 'dvd'     # traget domain: books, dvd, kitchen, or electronics


    ##Loading pre-embedded data
    print("Loading data ...")
    xs, ys, xt, yt, xtest, ytest = load_amazon(source_name, target_name, data_folder)

    #Define validation sets
    nb_valid = int(0.1 * len(ys))
    x_valid, y_valid = xs[-nb_valid:, :], ys[-nb_valid:]
    xs, ys = xs[0:-nb_valid, :], ys[0:-nb_valid]
    print xs.shape
    ys_hot = np.zeros((len(ys),2))
    ys_hot[ys] = 1
    yt_hot = np.zeros((len(yt),2))
    yt_hot[yt] = 1
    y_valid_hot = np.zeros((len(y_valid),2))
    y_valid_hot[y_valid] = 1
    ytest_hot = np.zeros((len(ytest),2))
    ytest_hot[ytest] = 1


    print("Training...")
    train(xs, ys_hot, xt, yt_hot, x_valid, y_valid_hot)

    print("Evaluating...")


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
