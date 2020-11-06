from ProcessEmbeddings import WordEmbeddings
import importlib
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow_fcwta.models import FullyConnectedWTA


def sparsify_embeddings(vector_file, target_dims, sparsity):
    tf.app.flags.DEFINE_float('learning_rate', 1e-2,
                              'learning rate to use during training')
    tf.app.flags.DEFINE_float('sparsity', 0.05,
                              'lifetime sparsity constraint to enforce')
    tf.app.flags.DEFINE_integer('train_size', 100,
                              'number of samples for train size')
    tf.app.flags.DEFINE_integer('batch_size', 256,
                                'batch size to use during training')
    tf.app.flags.DEFINE_integer('hidden_units', target_dims,
                                'size of each ReLU (encode) layer')
    tf.app.flags.DEFINE_integer('num_layers', 1,
                                'number of ReLU (encode) layers')
    tf.app.flags.DEFINE_integer('train_steps', 8000,
                                'total minibatches to train')
    tf.app.flags.DEFINE_integer('steps_per_display', 500,
                                'minibatches to train before printing loss')
    tf.app.flags.DEFINE_boolean('use_seed', True,
                                'fix random seed to guarantee reproducibility')
    tf.app.flags.DEFINE_boolean('show_plots', False,
                                'show visualizations')
    tf.app.flags.DEFINE_string('f', '', 'kernel')

    FLAGS = tf.app.flags.FLAGS

    fcwta = FullyConnectedWTA(300,
                              FLAGS.batch_size,
                              sparsity=FLAGS.sparsity,
                              hidden_units=FLAGS.hidden_units,
                              encode_layers=FLAGS.num_layers,
                              learning_rate=FLAGS.learning_rate)

    WE = WordEmbeddings(vector_file=vector_file)

    X_train = WE.embeds

    # Batches word embeds into groups of 256
    batch = tf.train.shuffle_batch(
            [X_train],
            batch_size= FLAGS.batch_size,
            capacity=512,
            min_after_dequeue=128,
            seed=1,
            enqueue_many=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        for step in range(FLAGS.train_steps):
            curr_batch = batch.eval(session=sess)
            _, loss = fcwta.step(sess, curr_batch)
            if step % FLAGS.steps_per_display == 0:
                print('step={}, loss={:.3f}'.format(step, loss))
        # Featurize data
        X_train_f = fcwta.encode(sess, X_train)
    return X_train_f
