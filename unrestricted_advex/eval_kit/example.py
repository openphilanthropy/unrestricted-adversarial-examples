"""Example on how to use the evaluation kit."""

import numpy as np


# our API are designed for binary problems, while using imagenet
# as an example, we simply convert the data into a binary problem
# of one-vs-rest for this class. The class is arbitrarily picked
# as this is just a demonstration of how the pipeline works.

IMAGENET_CLASS = 0


def get_model(model_key):
  if model_key == 'pytorch-pretrain':
    import torch
    import torchvision

    pytorch_model = torchvision.models.resnet50(pretrained=True)
    pytorch_model = pytorch_model.cuda()
    pytorch_model.eval()  # switch to eval mode

    def model_wrapper(x_np):
      with torch.no_grad():
        x = torch.from_numpy(x_np).cuda()
        logits1000 = pytorch_model(x)

        # model API needs a single probability in [0, 1]
        prob0 = torch.nn.functional.softmax(
            logits1000, dim=1)[:, IMAGENET_CLASS]
        return prob0.cpu().numpy()

    return model_wrapper

  elif model_key == 'keras-pretrain':
    import tensorflow as tf
    tf.keras.backend.set_image_data_format('channels_first')

    k_model = tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000)

    def model_wrapper(x_np):
      # it seems keras pre-trained model directly output softmax-ed probs
      prob1000 = k_model.predict_on_batch(x_np)
      return prob1000[:, IMAGENET_CLASS]

    return model_wrapper

  else:
    raise KeyError('Unknown model: {}'.format(model_key))


def get_data_iter(data_key):
  valdir = '/root/datasets/unpacked_imagenet_pytorch/val'
  batch_size = 128

  if data_key == 'pytorch-imagenet':
    import torch
    import torchvision
    import torchvision.transforms as transforms

    num_workers = 4
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]))
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size,
                                             shuffle=False, pin_memory=True,
                                             num_workers=num_workers)

    def loader_wrapper():
      for x_torch, y_torch in val_loader:
        yield (x_torch.numpy(),
               (y_torch.numpy() == IMAGENET_CLASS).astype(np.int))  # binary

    return loader_wrapper

  elif data_key == 'keras-imagenet':
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    tf.keras.backend.set_image_data_format('channels_first')

    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input)

    val_iter = val_gen.flow_from_directory(
        valdir, target_size=(224, 224), class_mode='sparse',
        batch_size=batch_size, shuffle=False)

    def loader_wrapper():
      for x_np, y_np in val_iter:
        yield x_np, (y_np == IMAGENET_CLASS).astype(np.int)

    return loader_wrapper

  else:
    raise KeyError('Unknown data iterator: {}'.format(data_key))


################################################################################
import argparse
from unrestricted_advex.eval_kit import eval_with_attacks


parser = argparse.ArgumentParser(description='Example evaluation')
parser.add_argument('--model', default='pytorch-pretrain',
                    choices=['pytorch-pretrain', 'keras-pretrain'])
parser.add_argument('--data', default='pytorch-imagenet',
                    choices=['pytorch-imagenet', 'keras-imagenet'])


if __name__ == '__main__':
  args = parser.parse_args()

  # right now we do not allow mixing the data pipeline and the model from
  # different framework. Tensorflow will try to claim all the GPU memory
  # at start by default; and cuDNN might also have some issue when being
  # called from 2 different frameworks, etc.
  if args.model.split('-')[0] != args.data.split('-')[0]:
    raise KeyError('Model ({}) and data pipeline ({}) should match'.format(
        args.model, args.data))

  model = get_model(args.model)
  data_iter = get_data_iter(args.data)
  attacks = ['null']

  # set max_num_batches to -1 to evaluate on the whole validation set
  labels, preds = eval_with_attacks.evaluate(model, data_iter, attacks,
                                             max_num_batches=10)

  print('{:>15s} | {}'.format('Attack', 'Accuracy'))
  for attack, preds_under_attack in zip(attacks, preds):
    acc = np.equal(labels, preds_under_attack > 0.5).astype(np.float64).mean()
    print('{:>15s} | {:.2f}%'.format(attack, acc * 100))
