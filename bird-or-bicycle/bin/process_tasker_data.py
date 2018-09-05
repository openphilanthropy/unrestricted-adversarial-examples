import csv
import json

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

  def _is_valid(self):
    if self._object_area() < MIN_BBOX_AREA:
      print("INVALID, object was too small: %s" % self.url)
      return False
      pass

    if self.is_depiction:
      print("INVALID, object was a depiction : %s" % self.url)
      return False

    if not ALLOW_OCCLUDED:
      if self.is_occluded:
        print("INVALID, object was occluded : %s" % self.url)
        return False

    if not ALLOW_TRUNCATED:
      if self.is_truncated:
        print("INVALID, object was truncated : %s" % self.url)
        return False

    return True

  def is_unambiguous_bird(self):
    return self._is_valid() and \
           self.bird_label == 'definitely_yes' and \
           self.bicycle_label == 'definitely_no'

  def is_unambiguous_bicycle(self):
    return self._is_valid() and \
           self.bird_label == 'definitely_no' and \
           self.bicycle_label == 'definitely_yes'


if __name__ == '__main__':

  unambiguous_bird_ids = []
  unambiguous_bird_urls = []

  unambiguous_bicycle_ids = []
  unambiguous_bicycle_urls = []

  filename = '/tmp/bird-or-bicycle-tasker-data.csv'
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    _ = next(reader)
    header = next(reader)
    for i, row in enumerate(reader):
      # print(i)
      # print(row)

      url, image_id = row[:2]

      tasker_1_data = row[2:9]
      tasker_2_data = row[9:16]
      tasker_3_data = row[16:23]

      tasker_responses = [
        TaskerResponse(tasker_1_data, url=url),
        TaskerResponse(tasker_2_data, url=url),
        TaskerResponse(tasker_3_data, url=url),
      ]

      if all([resp.is_unambiguous_bird() for resp in tasker_responses]):
        unambiguous_bird_ids.append(image_id)
        unambiguous_bird_urls.append(url)

      elif all([resp.is_unambiguous_bicycle() for resp in tasker_responses]):
        unambiguous_bicycle_ids.append(image_id)
        unambiguous_bicycle_urls.append(url)

      else:
        print("Ambiguous image: %s" % url)

  print('unambiguous_bicycle_ids: ', len(unambiguous_bicycle_ids))
  print('unambiguous_bird_ids: ', len(unambiguous_bird_ids))

  with open('/tmp/bird_image_ids.csv', 'w') as f:
    for bird_id in unambiguous_bird_ids:
      bird_id = bird_id[len('/bird/'):-len('.jpg')]

      f.write(bird_id + '\n')

  with open('/tmp/bicycle_image_ids.csv', 'w') as f:
    for bicycle_id in unambiguous_bicycle_ids:
      bicycle_id = bicycle_id[len('/bicycle/'):-len('.jpg')]

      f.write(bicycle_id + '\n')
