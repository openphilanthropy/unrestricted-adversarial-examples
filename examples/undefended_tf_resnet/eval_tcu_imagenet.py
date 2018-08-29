from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from absl import app as absl_app
from absl import flags

from examples.undefended_tf_resnet import BIRD_CLASSES, \
  BICYCLE_CLASSES
from examples.undefended_tf_resnet import EvalModeAttackableModel
from examples.undefended_tf_resnet import sess_with_loaded_model
from examples.undefended_tf_resnet import tcu_imagenet_pipeline

FLAGS = flags.FLAGS


def main(argv):
  del argv

  model = EvalModeAttackableModel()

  # TODO: add optional shuffling
  x_input, y_input = tcu_imagenet_pipeline.input_fn(
    shuffle=True).make_one_shot_iterator().get_next()

  with sess_with_loaded_model() as sess:
    # Sanity check

    accs = []
    trues = []

    for i in range(20):
      x_input_np, y_input_np = sess.run(
        [x_input, y_input])

      probs_np, = sess.run([model.probabilities],
                           feed_dict={
                             model.x_input: x_input_np,
                             model.y_input: y_input_np,
                           })

      bird_probs = np.max(probs_np[:, BIRD_CLASSES], axis=1)
      bicycle_probs = np.max(probs_np[:, BICYCLE_CLASSES], axis=1)

      pred_bicycle = bird_probs < bicycle_probs  # 1 if bicycle 0 if bird
      is_correct = (pred_bicycle) == y_input_np.astype(np.bool)

      batch_acc = np.mean(is_correct)
      print("fraction correct clean: ", batch_acc)
      accs.append(batch_acc)
      trues.append(y_input_np)

    print("\nAverage acc: ", np.mean(accs))
    print("\nAverage trues: ", np.mean(trues))

if __name__ == '__main__':
  absl_app.run(main)
