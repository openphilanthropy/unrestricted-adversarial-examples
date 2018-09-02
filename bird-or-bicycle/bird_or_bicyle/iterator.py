import bird_or_bicyle
import torch
import torchvision
from bird_or_bicyle import BICYCLE_IDX, BIRD_IDX


def get_iterator(split='train', batch_size=32, shuffle=True):
  """ Create a backend-agnostic iterator for the dataset.
  Images are formatted in channels-last in the Tensorflow style
  :param split: One of ['train', 'test', 'extras']
  :param batch_size: The number of images and labels in each batch
  :param shuffle: Whether or not to shuffle
  :return:  An iterable that returns (batched_images, batched_labels)
  """
  data_dir = bird_or_bicyle.get_dataset(split)

  train_dataset = torchvision.datasets.ImageFolder(
    data_dir,
    torchvision.transforms.Compose([
      torchvision.transforms.Resize(224),
      torchvision.transforms.ToTensor(),
      lambda x: torch.einsum('chw->hwc', [x]),
    ]))

  data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=shuffle)

  assert train_dataset.class_to_idx['bicycle'] == BICYCLE_IDX
  assert train_dataset.class_to_idx['bird'] == BIRD_IDX

  dataset_iter = [(x.numpy(), y.numpy()) for (x, y) in iter(data_loader)]
  return dataset_iter
