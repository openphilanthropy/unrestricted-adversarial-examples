import csv
import json
import os.path
from collections import defaultdict

MIN_BBOX_AREA = (0.4 * 299) ** 2

ALLOW_TRUNCATED = True
ALLOW_OCCLUDED = True


class TaskerResponse:
  def __init__(self, tasker_data, url):
    self._tasker_data = tasker_data
    worker_id, bird_label, bicycle_label, bounding_box_json, \
    is_not_truncated, is_not_occluded, is_not_depiction = tasker_data

    self.url = url

    self.worker_id = worker_id
    self.bird_label = bird_label
    self.bicycle_label = bicycle_label

    # Conservative, assume there is a problem
    # unless we explicitly know that there is no problem
    self.is_truncated = is_not_truncated != "TRUE"
    self.is_depiction = is_not_depiction != "TRUE"

    # NOTE: This was mislabeled in the headings
    self.is_occluded = is_not_occluded != "TRUE"

    self.bounding_box = json.loads(bounding_box_json)

  def _object_area(self):
    if not self.bounding_box or \
            'width' not in self.bounding_box or \
            'height' not in self.bounding_box:
      return 0

    return float(self.bounding_box['width']) * \
           float(self.bounding_box['height'])

  def has_large_object(self):
    return self._object_area() > MIN_BBOX_AREA

  def is_unambiguous_bird(self):
    return self.bird_label == 'definitely_yes' and \
           self.bicycle_label == 'definitely_no'

  def is_unambiguous_bicycle(self):
    return self.bird_label == 'definitely_no' and \
           self.bicycle_label == 'definitely_yes'


def _long_id_to_label_and_image_id(image_id):
  if image_id.startswith('/bird/'):
    return 'bird', image_id[len('/bird/'):-len('.jpg')]
  if image_id.startswith('/bicycle/'):
    return 'bicycle', image_id[len('/bicycle/'):-len('.jpg')]
  else:
    raise ValueError("Bad long_image_id: %s" % image_id)


# Images that are known to be mislabeled by taskers
KNOWN_BAD_IMAGES = {
  'e9310a5dce5bbb0f',
  '1422b0695f2d3d51',
  '1988a12b8997bab2',
  '3519bf7ab7d2f7b6',
  '599a814b35fb994d',
}

if __name__ == '__main__':

  unambiguous_bird_ids = []
  unambiguous_bird_urls = []

  unambiguous_bicycle_ids = []
  unambiguous_bicycle_urls = []

  rejection_reason_to_urls = defaultdict(list)

  # Raw tasker labels can be fetched with the following command:
  #
  # wget --no-check-certificate https://drive.google.com/open?id=1pfW-QEDKieQrEBioopGJxz-CJhrSiPJq -O /tmp/tasker_labels_0.0.4.csv
  #
  filename = '/tmp/tasker_labels_0.0.4.csv'
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    _ = next(reader)
    header = next(reader)
    for i, row in enumerate(reader):
      # print(i)
      # print(row)

      url, long_image_id = row[:2]
      orig_label, image_id = _long_id_to_label_and_image_id(long_image_id)
      assert len(image_id) == len('86d47eaee77d6e33')

      tasker_1_data = row[2:9]
      tasker_2_data = row[9:16]
      tasker_3_data = row[16:23]

      responses = [
        TaskerResponse(tasker_1_data, url),
        TaskerResponse(tasker_2_data, url),
        TaskerResponse(tasker_3_data, url),
      ]

      if image_id in KNOWN_BAD_IMAGES:
        rejection_reason_to_urls['known_bad'].append(url)
        continue

      if not all([resp.has_large_object() for resp in responses]):
        rejection_reason_to_urls['too_small'].append(url)
        continue

      if not all([not resp.is_depiction for resp in responses]):
        rejection_reason_to_urls['is_depiction'].append(url)
        continue

      if not ALLOW_OCCLUDED:
        if not all([not resp.is_occluded for resp in responses]):
          rejection_reason_to_urls['is_occluded'].append(url)
        continue

      if not ALLOW_TRUNCATED:
        if not all([not resp.is_truncated for resp in responses]):
          rejection_reason_to_urls['is_truncated'].append(url)
        continue

      if orig_label == 'bird' and \
          all([resp.is_unambiguous_bird() for resp in responses]):
        unambiguous_bird_ids.append(image_id)
        unambiguous_bird_urls.append(url)
      elif orig_label == 'bicycle' \
          and all([resp.is_unambiguous_bicycle() for resp in responses]):
        unambiguous_bicycle_ids.append(image_id)
        unambiguous_bicycle_urls.append(url)
      else:
        rejection_reason_to_urls['ambiguous'].append(url)
        # print("Ambiguous image: %s" % url)

  print('unambiguous_bicycle_ids: ', len(unambiguous_bicycle_ids))
  print('unambiguous_bird_ids: ', len(unambiguous_bird_ids))

  results_dir = '/tmp/'
  with open(os.path.join(results_dir, 'bird_image_ids.csv'), 'w') as f:
    for bird_id in unambiguous_bird_ids:
      f.write(bird_id + '\n')

  with open(os.path.join(results_dir, 'bicycle_image_ids.csv'), 'w') as f:
    for bicycle_id in unambiguous_bicycle_ids:
      f.write(bicycle_id + '\n')

  print("Wrote results to %s" % results_dir)
