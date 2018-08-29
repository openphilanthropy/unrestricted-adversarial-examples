from unrestricted_advex import eval_kit
from unrestricted_advex.load_models import get_keras_tcu_model

if __name__ == '__main__':
  undefended_keras_model = get_keras_tcu_model()
  eval_kit.evaluate_tcu_images_model(undefended_keras_model)
