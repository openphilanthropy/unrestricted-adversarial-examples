# Unrestricted Advex

Large mono-repo with all the projects used for the unrestricted-advex launch

Check inside individual folders for instructions.

## Structure

```
tcu_images
unrestricted_advex
  baselines
    images
      train_undefended_resnet.py
      eval_checkpoint.py (beats the null attack)
    mnist
      eval_checkpoint.py (beats the null attack)
      train_undefended.py
      train_madry.py
  eval_kit 
    evaluate_images_model(model) -> acc at 80% coverage
    evaluate_mnist_model(model) -> acc at 80% coverage
    attacks
      spsa etc
```
