import os.path

import bird_or_bicycle
import torch
import torchvision
from bird_or_bicycle import BICYCLE_IDX, BIRD_IDX


class ImageFolderWithFilenames(torchvision.datasets.ImageFolder):
  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (sample, target, image_id) where target is class_index of the target class.
    """
    path, _ = self.samples[index]
    sample, target = super(ImageFolderWithFilenames, self).__getitem__(index)
    image_id = os.path.basename(path).rstrip('.jpg')
    return sample, target, image_id


def get_iterator(split='train', batch_size=32, shuffle=True):
  """ Create a backend-agnostic iterator for the dataset.
  Images are formatted in channels-last in the Tensorflow style
  :param split: One of ['train', 'test', 'extras']
  :param batch_size: The number of images and labels in each batch
  :param shuffle: Whether or not to shuffle
  :return:  An iterable that returns (batched_images, batched_labels, batched_image_ids)
  """
  data_dir = bird_or_bicycle.get_dataset(split)

  image_preprocessing = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    lambda x: torch.einsum('chw->hwc', [x]),
  ])

  train_dataset = ImageFolderWithFilenames(
    data_dir, transform=image_preprocessing
  )

  data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=shuffle)

  assert train_dataset.class_to_idx['bicycle'] == BICYCLE_IDX
  assert train_dataset.class_to_idx['bird'] == BIRD_IDX

  dataset_iter = ((x_batch.numpy(), y_batch.numpy(), image_ids)
                  for (x_batch, y_batch, image_ids) in iter(data_loader))

  return dataset_iter
