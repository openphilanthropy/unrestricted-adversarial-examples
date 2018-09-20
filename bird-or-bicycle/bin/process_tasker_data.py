import csv
import json
from collections import defaultdict

MIN_BBOX_AREA = (0.5 * 299) ** 2

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


def _strip_name(image_id):
  if image_id.startswith('/bird/'):
    image_id = image_id[len('/bird/'):-len('.jpg')]
  if image_id.startswith('/bicycle/'):
    image_id = image_id[len('/bicycle/'):-len('.jpg')]

  assert len(image_id) == len('86d47eaee77d6e33')
  return image_id


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
  filename = '/Users/tomfeelslucky/Downloads/bird-or-bicycle-tasker-data - shuffled_urls_v0-0-1.csv'
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    _ = next(reader)
    header = next(reader)
    for i, row in enumerate(reader):
      # print(i)
      # print(row)

      url, long_image_id = row[:2]
      image_id = _strip_name(long_image_id)

      tasker_1_data = row[2:9]
      tasker_2_data = row[9:16]
      tasker_3_data = row[16:23]

      responses = [
        TaskerResponse(tasker_1_data, url=url),
        TaskerResponse(tasker_2_data, url=url),
        TaskerResponse(tasker_3_data, url=url),
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

      if all([resp.is_unambiguous_bird() for resp in responses]):
        unambiguous_bird_ids.append(image_id)
        unambiguous_bird_urls.append(url)
      elif all([resp.is_unambiguous_bicycle() for resp in responses]):
        unambiguous_bicycle_ids.append(image_id)
        unambiguous_bicycle_urls.append(url)
      else:
        rejection_reason_to_urls['ambiguous'].append(url)
        # print("Ambiguous image: %s" % url)

  print('unambiguous_bicycle_ids: ', len(unambiguous_bicycle_ids))
  print('unambiguous_bird_ids: ', len(unambiguous_bird_ids))

  with open('/tmp/bird_image_ids.csv', 'w') as f:
    for bird_id in unambiguous_bird_ids:
      f.write(bird_id + '\n')

  with open('/tmp/bicycle_image_ids.csv', 'w') as f:
    for bicycle_id in unambiguous_bicycle_ids:
      f.write(bicycle_id + '\n')
