import argparse

import tcu_images
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.utils import multi_gpu_model

parser = argparse.ArgumentParser(description='Keras ResNet50 ImageNet Training')

parser.add_argument('-d', '--tcu_images_data', metavar='DIR',
                    default="/tmp/data/tcu_images/",
                    help='path to dataset')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--model_dir', type=str, metavar='PATH',
                    help='path to model_dir',
                    default="/tmp/models/keras_undefended_resnet")

SIZE = (224, 224)


def _num_gpus():
  return len([d for d in device_lib.list_local_devices() if "GPU" in d.name])


def train():
  global args
  args = parser.parse_args()

  train_data_dir = tcu_images.get_dataset('train', verify=False)
  test_data_dir = tcu_images.get_dataset('test', verify=False)
  extras_data_dir = tcu_images.get_dataset('extras', verify=False)

  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.8,  # Choose a fraction for validation
  )

  gen_params = dict(
    target_size=SIZE,
    batch_size=args.batch_size,
    class_mode='binary',
  )

  train_generator = train_datagen.flow_from_directory(
    extras_data_dir, subset='training', **gen_params)

  validation_generator = train_datagen.flow_from_directory(
    extras_data_dir, subset='validation', **gen_params)

  # Validate using training data that is IID with test set
  test_datagen = ImageDataGenerator(rescale=1. / 255)
  test_val_generator = test_datagen.flow_from_directory(
    train_data_dir, **gen_params)
  test_generator = test_datagen.flow_from_directory(
    test_data_dir, **gen_params)


  model = tf.keras.applications.resnet50.ResNet50()
  # model.summary()

  if _num_gpus() > 1:
    model = multi_gpu_model(model, gpus=_num_gpus(), cpu_relocation=True)

  args.lr = 0.001 * (args.batch_size / 64)  # Scale up linearly with batch size

  print("=======================================")
  print("========== Starting training ==========")
  print("=======================================")
  print(args)

  model.compile(
    # keras.optimizers.Adam(lr=args.lr),
    keras.optimizers.SGD(lr=args.lr, momentum=0.9, nesterov=True),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

  reduce_lr = ReduceLROnPlateau(
    monitor='loss',
    patience=5,
    verbose=1,
    factor=0.5,
    min_lr=args.lr / 1000.)

  model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    epochs=args.epochs,
    use_multiprocessing=True,
    callbacks=[
      reduce_lr,
      keras.callbacks.ModelCheckpoint(
        "/tmp/imagenet_{epoch}.model",
        save_weights_only=True
      ),
    ],
    workers=int((args.batch_size / 16))
  )


if __name__ == '__main__':
  train()
