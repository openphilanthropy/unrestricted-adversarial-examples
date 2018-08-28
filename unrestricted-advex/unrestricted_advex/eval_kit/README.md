# Evaluation Interface

## Models

A model is a callable that takes a batched input image of shape (N, C, H, W),
and output a vector of length N, with each element in range [0, 1].

All the inputs and outputs should be numpy arrays. We work with only binary
classification problems. The value in [0, 1] is interpreted as the likelihood
of being class 1.

If the model needs to set up session, or eval mode, etc. It should be prepared
before passing into the evaluation code.

## Attacks

An attack is a callable that takes a model and a batched input (both specified
as above) as input, and output a batched adversarial examples, as numpy arrays.

Here we are using gradient free attacks in our evaluation, so the attacks can
evaluate the model but cannot get gradient information.


# Provided code

* `eval_with_attacks.py` provides function to evaluate a model with built-in
  attacks.
* `example.py` is an example on how to use the evaluation toolkit, by simply
  evaluating on pre-trained models.

To run the code with pytorch pipeline:
```
CUDA_VISIBLE_DEVICES=0 python example.py
```

Or to run the example code with keras pipeline:

```
CUDA_VISIBLE_DEVICES=0 python example.py --model=keras-pretrain --data=keras-imagenet
```

# TODOs

* Implement spatial attack
