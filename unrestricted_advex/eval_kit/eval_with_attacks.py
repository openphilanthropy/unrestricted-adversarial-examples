"""Evaluate a model with attacks."""

import numpy as np


def spsa_attack(model, batch_nchw, labels)::
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=(1,) + batch_shape[1:])
        y_label = tf.placeholder(tf.int32, shape=(1,))

        attack = SPSA(model)  # TODO cleverhansify
        x_adv = attack.generate(
            x_input, y=y_label, epsilon=epsilon, num_steps=30,
            early_stop_loss_threshold=-1., batch_size=32, spsa_iters=16,
            is_debug=True)

        logits = model.get_logits(x_adv)
        acc = _top_1_accuracy(logits, y_label)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        num_correct = 0.
        with tf.train.MonitoredSession(
                session_creator=session_creator) as sess:
            for i in xrange(num_images):
                acc_val = sess.run(acc, feed_dict={
                    x_input: np.expand_dims(images[i], axis=0),
                    y_label: np.expand_dims(labels[i], axis=0),
                })
                tf.logging.info('Accuracy: %s', acc_val)
                num_correct += acc_val
            assert (num_correct / num_images) < 0.1


def evaluate(model, data_iter, attacks=None, max_num_batches=1):
  if attacks is None:
    attacks = ['null']  # a single null attack

  all_labels = []
  all_preds = [[] for _ in attacks]

  for i_batch, (x_np, y_np) in enumerate(data_iter()):
    if max_num_batches > 0 and i_batch >= max_num_batches:
      break

    for attack_f, preds_container in zip(attacks, all_preds):
      if attack_f == 'null':
        x_adv = x_np
      else:
        x_adv = attack_f(model, x_np, y_np)

      y_pred = model(x_adv)
      y_pred = np.clip(y_pred, 0, 1)  # force into [0, 1]

      all_labels.append(y_np)
      preds_container.append(y_pred)

  all_labels = np.concatenate(all_labels)
  all_preds = [np.concatenate(x) for x in all_preds]

  return all_labels, all_preds
