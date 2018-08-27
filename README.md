# Unrestricted Advex

Repository of code accompanying the unrestricted-advex launch

Check inside individual folders for instructions.

## Structure

```
tcu_images

unrestricted_advex
  eval_kit 
    evaluate_images_model(model) -> acc at 80% coverage
    evaluate_mnist_model(model) -> acc at 80% coverage
    attacks
      spsa etc
  mnist_baselines
  resnet_baselines
```


## Warm-up Leaderboard

| Defense               | Submitted by  | SPSA acc@80% | Spatial acc@80% | Submission Date |
| --------------------- | ------------- | ------------ |--------------- | --------------- |
| [Undefended Baseline](https://github.com/google/unrestricted-adversarial-examples/tree/master/unrestricted_advex/pytorch_resnet_baseline)   |  --           |    **0%**    |     **0%**     |  Aug 27th, 2018 |