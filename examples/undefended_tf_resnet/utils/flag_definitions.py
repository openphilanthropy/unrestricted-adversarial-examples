"""Public interface for flag definition.

See _example.py for detailed instructions on defining flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from absl import app as absl_app
from absl import flags

# This codifies help string conventions and makes it easy to update them if
# necessary. Currently the only major effect is that help bodies start on the
# line after flags are listed. All flag definitions should wrap the text bodies
# with help wrap when calling DEFINE_*.
help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                              firstline_indent="\n")

# Replace None with h to also allow -h
absl_app.HelpshortFlag.SHORT_NAME = "h"


def set_defaults(**kwargs):
  for key, value in kwargs.items():
    flags.FLAGS.set_default(name=key, value=value)


def define_base():
  flags.DEFINE_string(
    name="data_dir", short_name="dd",
    default="/root/datasets/cloudtpu-imagenet-data/train",
    help=help_wrap("The location of the input data."))

  flags.DEFINE_string(
    name="model_dir", short_name="md", default="/tmp",
    help=help_wrap("The location of the model checkpoint files."))

  flags.DEFINE_string(
    name='eval_checkpoint_path',
    help='Path to load checkpoint from',
    default="/root/tb/imagenet/full-imagenet-72b3e7c/model.ckpt-61250")

  flags.DEFINE_boolean(
    name="clean", default=False,
    help=help_wrap("If set, model_dir will be removed if it exists."))

  flags.DEFINE_integer(
    name="train_epochs", short_name="te", default=1,
    help=help_wrap("The number of epochs used to train."))

  flags.DEFINE_integer(
    name="epochs_between_evals", short_name="ebe", default=1,
    help=help_wrap("The number of training epochs to run between "
                   "evaluations."))

  flags.DEFINE_float(
    name="stop_threshold", short_name="st",
    default=None,
    help=help_wrap("If passed, training will stop at the earlier of "
                   "train_epochs and when the evaluation metric is  "
                   "greater than or equal to stop_threshold."))

  flags.DEFINE_integer(
    name="batch_size", short_name="bs", default=32,
    help=help_wrap("Batch size for training and evaluation. When using "
                   "multiple gpus, this is the global batch size for "
                   "all devices. For example, if the batch size is 32 "
                   "and there are 4 GPUs, each GPU will get 8 examples on "
                   "each step."))

  flags.DEFINE_string(
    name="export_dir", short_name="ed", default=None,
    help=help_wrap("If set, a SavedModel serialization of the model will "
                   "be exported to this directory at the end of training. "
                   "See the README for more details and relevant links.")
  )


def define_device():
  """Register device specific flags."""

  flags.DEFINE_string(
    name="tpu", default=None,
    help=help_wrap(
      "The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a "
      "grpc://ip.address.of.tpu:8470 url. Passing `local` will use the"
      "CPU of the local instance instead. (Good for debugging.)"))

  flags.DEFINE_string(
    name="tpu_zone", default=None,
    help=help_wrap(
      "[Optional] GCE zone where the Cloud TPU is located in. If not "
      "specified, we will attempt to automatically detect the GCE "
      "project from metadata."))

  flags.DEFINE_string(
    name="tpu_gcp_project", default=None,
    help=help_wrap(
      "[Optional] Project name for the Cloud TPU-enabled project. If not "
      "specified, we will attempt to automatically detect the GCE "
      "project from metadata."))

  flags.DEFINE_integer(name="num_tpu_shards", default=8,
                       help=help_wrap("Number of shards (TPU chips)."))


def define_image():
  """Register image specific flags."""
  flags.DEFINE_enum(
    name="data_format", short_name="df", default='channels_last',
    enum_values=["channels_first", "channels_last"],
    help=help_wrap(
      "A flag to override the data format used in the model. "
      "channels_first provides a performance boost on GPU but is not "
      "always compatible with CPU. If left unspecified, the data format "
      "will be chosen automatically based on whether TensorFlow was "
      "built for CPU or GPU."))


# Map string to (TensorFlow dtype, default loss scale)
DTYPE_MAP = {
  "fp16": (tf.float16, 128),
  "fp32": (tf.float32, 1),
}


def get_tf_dtype(flags_obj):
  return DTYPE_MAP[flags_obj.dtype][0]


def get_loss_scale(flags_obj):
  if flags_obj.loss_scale is not None:
    return flags_obj.loss_scale
  return DTYPE_MAP[flags_obj.dtype][1]


def define_performance():
  """Register flags for specifying performance tuning arguments. """

  flags.DEFINE_integer(
    name="inter_op_parallelism_threads", short_name="inter", default=0,
    help=help_wrap("Number of inter_op_parallelism_threads to use for CPU. "
                   "See TensorFlow config.proto for details.")
  )

  flags.DEFINE_integer(
    name="intra_op_parallelism_threads", short_name="intra", default=0,
    help=help_wrap("Number of intra_op_parallelism_threads to use for CPU. "
                   "See TensorFlow config.proto for details."))

  flags.DEFINE_bool(
    name="use_synthetic_data", short_name="synth", default=False,
    help=help_wrap(
      "If set, use fake data (zeroes) instead of a real dataset. "
      "This mode is useful for performance debugging, as it removes "
      "input processing steps, but will not learn anything."))

  flags.DEFINE_integer(
    name="max_train_steps", short_name="mts", default=None, help=help_wrap(
      "The model will stop training if the global_step reaches this "
      "value. If not set, training will run until the specified number "
      "of epochs have run as usual. It is generally recommended to set "
      "--train_epochs=1 when using this flag."
    ))

  # flags.DEFINE_integer(
  #   name="save_checkpoint_steps", short_name="scs", default=None, help=help_wrap(
  #     "The frequency to save checkpoints. Defaults to once every 10 mins"
  #   ))

  flags.DEFINE_enum(
    name="dtype", short_name="dt", default="fp32",
    enum_values=DTYPE_MAP.keys(),
    help=help_wrap("The TensorFlow datatype used for calculations. "
                   "Variables may be cast to a higher precision on a "
                   "case-by-case basis for numerical stability."))

  flags.DEFINE_integer(
    name="loss_scale", short_name="ls", default=None,
    help=help_wrap(
      "The amount to scale the loss by when the model is run. Before "
      "gradients are computed, the loss is multiplied by the loss scale, "
      "making all gradients loss_scale times larger. To adjust for this, "
      "gradients are divided by the loss scale before being applied to "
      "variables. This is mathematically equivalent to training without "
      "a loss scale, but the loss scale helps avoid some intermediate "
      "gradients from underflowing to zero. If not provided the default "
      "for fp16 is 128 and 1 for all other dtypes."))

  loss_scale_val_msg = "loss_scale should be a positive integer."

  @flags.validator(flag_name="loss_scale", message=loss_scale_val_msg)
  def _check_loss_scale(loss_scale):  # pylint: disable=unused-variable
    if loss_scale is None:
      return True  # null case is handled in get_loss_scale()

    return loss_scale > 0

  flags.DEFINE_string(
    name="all_reduce_alg", short_name="ara", default=None,
    help=help_wrap("Defines the algorithm to use for performing all-reduce."
                   "See tf.contrib.distribute.AllReduceCrossTowerOps for "
                   "more details and available options."))


def define_resnet():
  flags.DEFINE_enum(
    name='resnet_version', short_name='rv', default='2',
    enum_values=['1', '2'],
    help=help_wrap(
      'Version of ResNet. (1 or 2) See README.md for details.'))

  flags.DEFINE_enum(enum_values=['18', '34', '50', '101', '152', '200'],
                    name='resnet_size', short_name='rs', default='50',
                    help=help_wrap(
                      'The size of the ResNet model to use.'))

  # The current implementation of ResNet v1 is numerically unstable when run
  # with fp16 and will produce NaN errors soon after training begins.
  msg = ('ResNet version 1 is not currently supported with fp16. '
         'Please use version 2 instead.')

  @flags.multi_flags_validator(['dtype', 'resnet_version'], message=msg)
  def _forbid_v1_fp16(flag_values):  # pylint: disable=unused-variable
    return (DTYPE_MAP[flag_values['dtype']][0] != tf.float16 or
            flag_values['resnet_version'] != '1')
