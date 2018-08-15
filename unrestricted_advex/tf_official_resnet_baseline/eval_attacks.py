from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from absl import app as absl_app
from absl import flags
from tensorflow.python.training import monitored_session
from unrestricted_advex.tf_official_resnet_baseline.imagenet_main import input_fn, get_num_gpus
from unrestricted_advex.tf_official_resnet_baseline.spatial_attack import \
  SpatialGridAttack, EvalModeAttackableModel
from unrestricted_advex.tf_official_resnet_baseline.utils.misc import distribution_utils

FLAGS = flags.FLAGS


def sess_with_loaded_model():
  session_creator = monitored_session.ChiefSessionCreator(
    checkpoint_filename_with_path=FLAGS.eval_checkpoint_path,
    master='')
  return monitored_session.MonitoredSession(session_creator=session_creator)


def main(argv):
  del argv

  input_fn_eval = input_fn(
    is_training=False,
    data_dir=FLAGS.data_dir,
    batch_size=distribution_utils.per_device_batch_size(
      FLAGS.batch_size, get_num_gpus()),
    num_epochs=1,
    repeat_single_batch=FLAGS.repeat_single_batch)

  x_input_gen, y_input_gen = input_fn_eval.make_one_shot_iterator().get_next()

  model = EvalModeAttackableModel()
  null_attack = SpatialGridAttack(
    model,
    spatial_limits=[0, 0, 0],
  )
  spatial_attack = SpatialGridAttack(
    model,
    spatial_limits=[3, 3, 30],
    # grid_granularity=[3, 3, 3],
  )

  with sess_with_loaded_model() as sess:
    for i in range(2):
      x_input_np, y_input_np = sess.run([x_input_gen, y_input_gen])

      # Sanity check
      probs_np, = sess.run([model.probabilities],
                           feed_dict={
                             model.x_input: x_input_np,
                             model.y_input: y_input_np,
                           })

      correct = np.equal(np.argmax(probs_np, axis=1), y_input_np)
      print("fraction correct clean: ", np.mean(correct))

      # Adversarial spatial transformations
      # Spatial transformation allowed
      worst_x_np, worst_transform_np = spatial_attack.perturb(
        x_nat=x_input_np,
        y_sparse=y_input_np,
        sess=sess
      )

      probs_np, = sess.run([model.probabilities],
                           feed_dict={
                             model.x_input: worst_x_np,
                             model.y_input: y_input_np,
                           })

      correct = np.equal(np.argmax(probs_np, axis=1), y_input_np)
      print("fraction correct adv, real trans: ", np.mean(correct))

      # Adversarial spatial transformations
      # Zero actual transformation allowed
      worst_x_np, worst_transform_np = null_attack.perturb(
        x_nat=x_input_np,
        y_sparse=y_input_np,
        sess=sess
      )

      probs_np, = sess.run([model.probabilities],
                           feed_dict={
                             model.x_input: worst_x_np,
                             model.y_input: y_input_np,
                           })

      correct = np.equal(np.argmax(probs_np, axis=1), y_input_np)
      print("fraction correct adv, no trans: ", np.mean(correct))

      print('\n\n')


if __name__ == '__main__':
  absl_app.run(main)
