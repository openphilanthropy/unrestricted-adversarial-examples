# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
import logging
import multiprocessing
import os
import subprocess
from multiprocessing import Pool
from subprocess import check_output

import torchvision
from PIL import Image
from bird_or_bicycle import metadata
from tqdm import tqdm

VERSION = '0.0.3'
METADATA_ROOT = os.path.dirname(metadata.__file__)

# https://stackoverflow.com/questions/20886565/using-multiprocessing-process-with-a-maximum-number-of-simultaneous-processes
N_WORKERS = multiprocessing.cpu_count() * 2 or 1

OPEN_IMAGES_BIRD_CLASS = '/m/015p6'
OPEN_IMAGES_BICYCLE_CLASS = '/m/0199g'

CLASS_NAME_TO_IMAGENET_CLASS = {
  'bird': list(range(80, 100 + 1)),
  'bicycle': [671, 444]
}

BICYCLE_IDX = 0
BIRD_IDX = 1


def _is_valid_extras_image(bbox_row, strict=False, min_bbox_area=0.2 ** 2):
  """
  :param bbox_row: A single row from the bbox.csv file
  :param strict:
  :param min_bbox_area:
  :return:
  """
  image_id, source, label_name, confidence, x_min, x_max, y_min, y_max, is_occluded, is_truncated, is_group_of, is_depiction, is_inside = bbox_row

  if strict:
    if int(is_occluded) or int(is_depiction) or int(is_inside) or int(
        is_group_of) or int(is_truncated):
      return False

  # Check that the image is not very small
  x_min, x_max, y_min, y_max = float(x_min), float(x_max), float(y_min), float(
    y_max)
  width = x_max - x_min
  height = y_max - y_min

  # Require that the object takes up at least 1/5th by 1/5th of the image
  area = width * height
  return area > min_bbox_area


def _get_extras_image_ids():
  """Return a map from label to set of image ids"""
  # Fetch metadata
  bbox_url = 'https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv'
  bbox_file = os.path.join(METADATA_ROOT, 'train-annotations-bbox.csv')
  if not os.path.isfile(bbox_file):
    cmd = 'wget %s -O %s' % (bbox_url, bbox_file)
    print(cmd)
    check_output(cmd, shell=True)

  print("Finding all birds and bicycles within OpenImages training dataset...")
  label_to_extras = {
    'bird': set(),
    'bicycle': set(),
  }

  with open(bbox_file, 'r') as f:
    bbox_reader = csv.reader(f)

    bbox_header = next(bbox_reader)  # skip the header
    logging.info(bbox_header)

    for bbox_row in tqdm(bbox_reader, total=14610230):
      # Unpack the bbox_line
      image_id, source, label_id, confidence, x_min, \
      x_max, y_min, y_max, is_occluded, is_truncated, \
      is_group_of, is_depiction, is_inside = bbox_row

      if not _is_valid_extras_image(bbox_row):
        continue

      if label_id == OPEN_IMAGES_BICYCLE_CLASS:
        label_to_extras['bicycle'].add(image_id)
      if label_id == OPEN_IMAGES_BIRD_CLASS:
        label_to_extras['bird'].add(image_id)

  print("Removing any images that show up in train or test set...")
  for split in ['train', 'test']:
    label_to_used_images = _get_bird_and_bicycle_image_ids(split)
    for label_name in ['bird', 'bicycle']:
      label_to_extras[label_name] -= label_to_used_images[label_name]

  # Truncate to the first 13000 images
  num_extras = metadata.NUM_IMAGES_PER_CLASS[VERSION]['extras']
  for label_name in ['bird', 'bicycle']:
    print(len(label_to_extras[label_name]))
    label_to_extras[label_name] = set(sorted(label_to_extras[label_name])[:num_extras])

  return label_to_extras


def _download_image(src_and_dest_dir):
  src, dest_dir = src_and_dest_dir
  cmd = "aws s3 --no-sign-request cp %s %s" % (src, dest_dir)
  try:
    check_output(cmd, shell=True)
  except subprocess.CalledProcessError:
    pass


def _map_with_tqdm(fn, iterable, n_workers=N_WORKERS, total=None):
  pool = Pool(n_workers)

  # Multiprocess map with a tqdm progress bar
  # https://github.com/tqdm/tqdm/issues/484#issuecomment-351001534
  for _ in tqdm(pool.imap_unordered(fn, iterable), total=total):
    pass  # Process work
  pool.close()
  pool.join()


def _download_to_dir(image_ids, dest_dir, split):
  print("Version: {VERSION}. Saving {n_images} images to {dest_dir} \
    (using {N_WORKERS} parallel processes)".format(
    VERSION=VERSION,
    n_images=len(image_ids),
    dest_dir=dest_dir,
    N_WORKERS=N_WORKERS))

  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

  srcs = ["s3://open-images-dataset/train/%s.jpg" % (image_id)
          for image_id in image_ids]
  srcs_and_dest_dirs = zip(srcs, [dest_dir] * len(srcs))

  _map_with_tqdm(_download_image, srcs_and_dest_dirs,
                 total=len(image_ids))


def _compute_sha1sum_of_directory(dir):
  print("Checking sha1sum of %s" % dir)
  cmd = "cd %s && find . | grep .jpg | sha1sum" % dir
  return check_output(cmd, shell=True).decode('utf-8')[:40]


def _resize_and_centercrop_image(image_path):
  image = Image.open(image_path)

  image = torchvision.transforms.Resize(299)(image)
  image = torchvision.transforms.CenterCrop(299)(image)
  image.save(image_path)


def _crop_and_resize_images(split_root):
  print("Cropping images to correct shape and size...")
  for label_name in ['bird', 'bicycle']:
    class_dir = os.path.join(split_root, label_name)
    images_in_class = os.listdir(class_dir)
    image_paths = [os.path.join(class_dir, image_name) for image_name in images_in_class]
    _map_with_tqdm(_resize_and_centercrop_image, image_paths,
                   total=len(image_paths))


def _get_bird_and_bicycle_image_ids(split):
  """Return a map from label to set of image ids"""
  label_name_to_image_ids = {}
  for label_name in ['bird', 'bicycle']:
    path = os.path.join(METADATA_ROOT, VERSION, "%s_image_ids.csv" % label_name)
    with open(path, 'r') as f:
      image_ids = f.read().strip().split('\n')
      if split == "test":
        image_ids = image_ids[0:500]
      elif split == 'train':
        image_ids = image_ids[500:1000]
      else:
        raise ValueError()

      label_name_to_image_ids[label_name] = set(image_ids)
  return label_name_to_image_ids


def default_data_root():
  return os.path.join(os.path.expanduser('~/datasets/bird_or_bicycle'), VERSION)


def verify_dataset_integrity(split, data_root=None):
  """Check the sha1sum of image names and the size of all images
  to make sure we have prepared the dataset correctly"""
  if data_root is None:
    data_root = default_data_root()

  split_root = os.path.join(data_root, split)
  for label_name in ['bird', 'bicycle']:
    class_dir = os.path.join(split_root, label_name)
    images_in_class = os.listdir(class_dir)
    expected_images = metadata.NUM_IMAGES_PER_CLASS[VERSION][split]

    for image_name in images_in_class:
      image_path = os.path.join(class_dir, image_name)
      image = Image.open(image_path)
      assert image.size == (299, 299), "Found image of wrong size at %s. \
      Wanted (299,299) but got %s" % (image_path, image.size)

    assert len(images_in_class) == expected_images, \
      "Incomplete dataset in %s: Expected %s images and found %s. \
      Please remove the corrupt dataset and try again" % (
        class_dir, expected_images, len(images_in_class))

  if False:  # Disable checksum for now
    shasum = _compute_sha1sum_of_directory(split_root)
    assert shasum == metadata.SHASUMS[
      VERSION][split], "sha1sum mismatch (got: %s). Please remove the files in %s" % (
      shasum, split_root)
  print("Verification of dataset successful. Dataset is correctly prepared.")


def get_dataset(split, data_root=None, force_download=False, verify=True):
  if data_root is None:
    data_root = default_data_root()
  split_root = os.path.join(data_root, split)

  if os.path.isdir(split_root) and not force_download:
    if verify:
      verify_dataset_integrity(split, data_root)
    return split_root

  if split == 'train':
    label_name_to_image_ids = _get_bird_and_bicycle_image_ids(split)
  elif split == 'test':
    label_name_to_image_ids = _get_bird_and_bicycle_image_ids(split)
  elif split == 'extras':
    label_name_to_image_ids = _get_extras_image_ids()
  else:
    raise ValueError("Split must be one of ['train', 'test', 'extras']")

  for label_name in ['bird', 'bicycle']:
    image_ids = label_name_to_image_ids[label_name]
    dest_dir = os.path.join(split_root, label_name)
    _download_to_dir(image_ids, dest_dir, split)

  _crop_and_resize_images(split_root)
  if verify:
    verify_dataset_integrity(split, data_root)
  return split_root
