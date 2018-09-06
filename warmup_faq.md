# Unrestricted Adversarial Examples Warm-up FAQ

#### How did you decide which attacks to use for the warm-up?

We wanted to choose attacks with the following properties:

- A well-tested canonical implementation is available
- Gradient-free, to avoid the problem of obfuscated gradients
- Fairly computationally efficient  

Beyond that, we focused on attacks that cover a variety of neighborhoods beyond the typical L_infinity ball 

#### How did you decide what hyperparameters to use for the attacks?

We want our `eval_kit` to complete within 24 hours on a single P100 GPU. This allows a typical university laboratory to run many evaluations before publishing their results.

Given that constraint on total compute, we want attacks to be as strong as possible. The compute allocation of the current `eval_kit` is something like 45% SPSA, 45% Boundary, 10% spatial. 

