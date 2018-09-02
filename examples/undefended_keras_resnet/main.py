import bird_or_bicyle
from unrestricted_advex import eval_kit
from unrestricted_advex.load_models import get_keras_bird_or_bicycle_model

if __name__ == '__main__':
  undefended_keras_model = get_keras_bird_or_bicycle_model()

  eval_kit.evaluate_bird_or_bicycle_model(
    undefended_keras_model,
    dataset_iter=bird_or_bicyle.get_iterator('train'))
