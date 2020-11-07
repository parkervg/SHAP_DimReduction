import tensorflow.compat.v1 as tf

tf.app.flags.DEFINE_float("learning_rate", 1e-2, "learning rate to use during training")
tf.app.flags.DEFINE_float("sparsity", 0.05, "lifetime sparsity constraint to enforce")
tf.app.flags.DEFINE_integer("train_size", 100, "number of samples for train size")
tf.app.flags.DEFINE_integer("batch_size", 256, "batch size to use during training")
tf.app.flags.DEFINE_integer("hidden_units", 3000, "size of each ReLU (encode) layer")
tf.app.flags.DEFINE_integer("num_layers", 1, "number of ReLU (encode) layers")
tf.app.flags.DEFINE_integer(
    "steps_per_checkpoint", 8000, "minibatches to train before saving checkpoint"
)
tf.app.flags.DEFINE_integer("train_steps", 8000, "total minibatches to train")
tf.app.flags.DEFINE_integer(
    "steps_per_display", 500, "minibatches to train before printing loss"
)
tf.app.flags.DEFINE_boolean(
    "use_seed", True, "fix random seed to guarantee reproducibility"
)
tf.app.flags.DEFINE_boolean("show_plots", False, "show visualizations")
tf.app.flags.DEFINE_string(
    "train_dir",
    "tensorflow_fcwta",
    "where to store checkpoints to (or load checkpoints from)",
)
tf.app.flags.DEFINE_string("f", "", "kernel")

FLAGS = tf.app.flags.FLAGS
