"""Evaluate a model with attacks."""

from six.moves import xrange
from cleverhans.model import Model
from cleverhans.attacks import SPSA
import numpy as np
import tensorflow as tf
import torch
import torchvision


import tcu_images


class CleverhansModelWrapper(Model):
    def __init__(self, model_fn):
        """
        Wrap a callable function that takes a numpy array of shape (N, C, H, W),
        and outputs a numpy vector of length N, with each element in range [0, 1].
        """
        self.nb_classes = 2
        def two_class_model_fn(x):
            class_one_logit = model_fn(x)
            class_zero_logit = -class_one_logit
            stack = np.vstack([class_zero_logit, class_one_logit]).T
            return stack
        self.model_fn = two_class_model_fn

    def fprop(self, x, **kwargs):
        logits_op = tf.py_func(self.model_fn, [x], tf.float32)
        return {'logits': logits_op}


def spsa_attack(model, batch_nchw, labels, epsilon=(4. / 255)):
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=(1,) + batch_nchw.shape[1:])
        y_label = tf.placeholder(tf.int32, shape=(1,))

        cleverhans_model = CleverhansModelWrapper(model)
        attack = SPSA(cleverhans_model)
        x_adv = attack.generate(
            x_input, y=y_label, epsilon=epsilon, num_steps=30,
            early_stop_loss_threshold=-1., batch_size=batch_nchw.shape[0],
            spsa_iters=16, is_debug=True)

        # Run computation
        with tf.Session() as sess:
            for i in xrange(batch_nchw.shape[0]):
                x_adv_np = sess.run(x_adv, feed_dict={
                    x_input: np.expand_dims(batch_nchw[i], axis=0),
                    y_label: np.expand_dims(labels[i], axis=0),
                })
        return x_adv_np


def evaluate(model, data_iter, attacks=None, max_num_batches=1):
  if attacks is None:
    attacks = ['null']  # a single null attack

  all_labels = []
  all_preds = [[] for _ in attacks]

  for i_batch, (x_np, y_np) in enumerate(data_iter):
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


BIRD_CLASSES = list(range(80,100+1))
BICYCLE_CLASSES = [671, 444]


def main():
  ### Get data
  data_dir = tcu_images.get_dataset('train')

  normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  train_dataset = torchvision.datasets.ImageFolder(
    data_dir,
    torchvision.transforms.Compose([
      torchvision.transforms.Resize(224),
      torchvision.transforms.ToTensor(),
      normalize,
    ]))

  dataset_iter = [(x.numpy(), y.numpy())
                  for (x, y) in iter(torch.utils.data.DataLoader(train_dataset, batch_size=32))]

  ### Load model
  pytorch_model = torchvision.models.resnet50(pretrained=True)
  pytorch_model = pytorch_model.cuda()
  pytorch_model.eval()  # switch to eval mode

  def model_fn(x_np):
    with torch.no_grad():
      x = torch.from_numpy(x_np).cuda()
      logits1000 = pytorch_model(x)

      # model API needs a single probability in [0, 1]
      bird_max_logit, _ = torch.max(logits1000[:, BIRD_CLASSES], dim=1)
      bicycle_max_logit, _ = torch.max(logits1000[:, BICYCLE_CLASSES], dim=1)
      delta_logits = bird_max_logit - bicycle_max_logit

      return delta_logits.cpu().numpy()

  ### Evaluate attack
  all_labels, all_preds = evaluate(model_fn, dataset_iter, attacks=[spsa_attack], max_num_batches=1)
  correct = np.sum(np.equal(all_preds[0], all_labels).astype(np.float32)) / len(all_labels)
  print(correct)

if __name__ == '__main__':
  main()
