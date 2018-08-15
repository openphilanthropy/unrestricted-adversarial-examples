"""Evaluate a model with attacks."""

import numpy as np




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
