"""Contains utility and supporting functions for ResNet on ImageNet.

This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf  # pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
from cleverhans.attacks import MadryEtAl
from cleverhans.model import CallableModelWrapper
from undefended_tf_resnet import official_resnet_model
from undefended_tf_resnet.official_imagenet_input_pipeline import input_fn, \
  _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, \
  _NUM_IMAGES
from undefended_tf_resnet.official_resnet_model import ImagenetModel
from undefended_tf_resnet.utils import flag_definitions

from baselines.undefended_tf_resnet.utils.misc import distribution_utils
from baselines.undefended_tf_resnet.utils.misc import model_helpers

flag_definitions.define_base()
flag_definitions.define_performance()
flag_definitions.define_image()
flag_definitions.define_resnet()

flags.DEFINE_boolean(name="eval_only", help="Skip training and do evaluation",
                     default=False)
flags.DEFINE_boolean(name="repeat_single_batch",
                     help="Sanity check that we can overfit a single batch",
                     default=False)

flags.DEFINE_boolean(name="use_pgd_attack", default=False,
                     help="Use PGD attack")

flag_definitions.set_defaults(
  train_epochs=100,
  dtype='fp16',  # Default to fp16 for faster training
  epochs_between_evals=1,
)
IM_SHAPE = [_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS]
FLAGS = flags.FLAGS


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = 0.1 * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn


def model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  learning_rate_fn = learning_rate_with_decay(
    batch_size=params['batch_size'], batch_denom=256,
    num_images=_NUM_IMAGES['train'],
    boundary_epochs=[30, 60, 80, 90],
    decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

  return resnet_model_fn(
    features=features,
    labels=labels,
    mode=mode,
    resnet_size=params['resnet_size'],
    weight_decay=1e-4,
    learning_rate_fn=learning_rate_fn,
    momentum=0.9,
    data_format=params['data_format'],
    resnet_version=params['resnet_version'],
    loss_scale=params['loss_scale'],
    dtype=params['dtype'],
    use_pgd_attack=params['use_pgd_attack'],
  )


def resnet_model_fn(features, labels, mode,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale,
                    dtype=official_resnet_model.DEFAULT_DTYPE,
                    use_pgd_attack=False):
  """Shared functionality for different tf_official_resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    dtype: the TensorFlow dtype to use for calculations.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # assert len(labels.get_shape()) == 1, "Labels must be flat"

  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  images = tf.cast(features, dtype)

  callable_model = ImagenetModel(resnet_size, data_format,
                                 resnet_version=resnet_version,
                                 dtype=dtype)
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  if use_pgd_attack:
    logits = callable_model(images, training=is_training)

    # Always generate the attack with training False to avoid updating BN
    cleverhans_model = CallableModelWrapper(
      lambda x: callable_model(x, training=False),
      'logits')
    attack = MadryEtAl(cleverhans_model)

    adv_images = attack.generate(
      images,
      eps=8.0,  # 8/255 = 0.031 (standard ImageNet eval)
      eps_iter=100
    )
    adv_logits = callable_model(tf.stop_gradient(adv_images),
                                training=is_training)
  else:
    logits = callable_model(images, training=is_training)

    # Just use the regular logits and images when doing clean training
    adv_images = images
    adv_logits = logits

  num_summary_images = 16
  tf.summary.image('images', images[0:num_summary_images])
  tf.summary.image('adv_images', adv_images[0:num_summary_images])

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32, name='logits')

  accuracy_clean = batch_accuracy(labels, logits, name='batch_accuracy_clean')
  tf.summary.scalar('batch_accuracy_clean', accuracy_clean)

  accuracy_adv = batch_accuracy(labels, adv_logits, name='batch_accuracy_adv')
  tf.summary.scalar('batch_accuracy_adv', accuracy_adv)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
    logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  predictions = {
    'logits': logits,
    'predictions': tf.argmax(logits, axis=1),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    'adv_images': adv_images,
    'images': images,
    'cross_entropy': cross_entropy,
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      export_outputs={
        'predict': tf.estimator.export.PredictOutput(predictions)
      })

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
    # loss is computed using fp32 for numerical stability.
    [tf.nn.l2_loss(tf.cast(v, tf.float32))
     for v in tf.trainable_variables()
     if 'batch_normalization' not in v.name])  # Omit batch_norm from WD
  tf.summary.scalar('l2_loss', tf.reduce_mean(l2_loss))
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    tf.identity(global_step, name='step')

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=momentum
    )

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      minimize_op = optimizer.minimize(loss, global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
    metrics = {}

  else:  # mode == EVAL
    train_op = None
    # Metrics are currently not compatible with distribution strategies
    # during training, so we only do it during eval
    # We can use the batch_accuracy summaries to get a rough idea of perf

    metrics = {
      'accuracy': tf.metrics.accuracy(labels, tf.argmax(logits, axis=1)),
      'adv_accuracy': tf.metrics.accuracy(labels, tf.argmax(adv_logits, axis=1))
    }

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=metrics)


def batch_accuracy(labels, logits, name='batch_accuracy'):
  correct = tf.equal(tf.to_int32(tf.argmax(logits, axis=1)), labels)
  acc = tf.reduce_mean(tf.to_float(correct), axis=0)
  return tf.identity(acc, name=name)


def get_num_gpus():
  """Treat num_gpus=-1 as 'use all'."""
  from tensorflow.python.client import \
    device_lib  # pylint: disable=g-import-not-at-top
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])


def main(_):
  model_helpers.apply_clean(flags.FLAGS)

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Create session config based on values of inter_op_parallelism_threads and
  # intra_op_parallelism_threads. Note that we default to having
  # allow_soft_placement = True, which is required for multi-GPU and not
  # harmful for other modes.
  session_config = tf.ConfigProto(
    inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
    intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
    allow_soft_placement=True)

  distribution_strategy = distribution_utils.get_distribution_strategy(
    get_num_gpus(), FLAGS.all_reduce_alg)

  run_config = tf.estimator.RunConfig(
    train_distribute=distribution_strategy,
    session_config=session_config,
  )

  classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config,
    params={
      'resnet_size': int(FLAGS.resnet_size),
      'data_format': FLAGS.data_format,
      'batch_size': FLAGS.batch_size,
      'resnet_version': int(FLAGS.resnet_version),
      'loss_scale': flag_definitions.get_loss_scale(FLAGS),
      'dtype': flag_definitions.get_tf_dtype(FLAGS),
      'use_pgd_attack': FLAGS.use_pgd_attack,
    })

  log_name_to_tensor = {x: x for x in [
    'batch_accuracy_clean',
    'batch_accuracy_adv',
    'cross_entropy',
    'learning_rate',
    'step',
  ]}
  hooks = [
    tf.train.LoggingTensorHook(
      tensors=log_name_to_tensor,
      every_n_iter=20
    )]

  def input_fn_train():
    return input_fn(
      is_training=True, data_dir=FLAGS.data_dir,
      batch_size=distribution_utils.per_device_batch_size(
        FLAGS.batch_size, get_num_gpus()),
      num_epochs=FLAGS.epochs_between_evals,
      num_gpus=get_num_gpus(),
      repeat_single_batch=FLAGS.repeat_single_batch,
    )

  def input_fn_eval():
    return input_fn(
      is_training=False, data_dir=FLAGS.data_dir,
      batch_size=distribution_utils.per_device_batch_size(
        FLAGS.batch_size, get_num_gpus()),
      num_epochs=1)

  total_training_cycle = (FLAGS.train_epochs //
                          FLAGS.epochs_between_evals)
  for cycle_index in range(total_training_cycle):
    tf.logging.info('Starting a training cycle: %d/%d',
                    cycle_index, total_training_cycle)

    if not FLAGS.eval_only:
      classifier.train(input_fn=input_fn_train, hooks=hooks,
                       max_steps=FLAGS.max_train_steps)

    tf.logging.info('Starting to evaluate.')
    eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                       steps=100)

    if model_helpers.past_stop_threshold(
        FLAGS.stop_threshold, eval_results['accuracy']):
      break


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  absl_app.run(main)
