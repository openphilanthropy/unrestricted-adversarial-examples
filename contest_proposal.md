# Unrestricted Adversarial Examples Contest Proposal

(TODO: copy content from paper into this section immediately before publication)

# Contest Prize Distribution Scheme
(TODO: copy content from paper into this section immediately before publication)

# Contest Mechanics FAQ
(TODO: copy content from paper into this section immediately before publication)

#### Why run an ongoing contest?

An ongoing contest has four main advantages:

1. A well-funded contest lets the world know that this problem is conspicuously open, and could get more researchers to see this as an important area.
2. An active, ongoing contest will decrease the cycle-time between “promising idea for a defense” and “consensus that it works or doesn’t work”. 
3. Submitted attacks and defenses will be required to be open source and to conform to a standard API, lowering the barrier to entry for new researchers.
4. The research community can monitor the current state of attacks vs defenses, and watch progress (or note the lack thereof).



#### Why do we need a model that *never* makes a mistake? Why can’t we give partial credit to defenders in the contest?
One advantage of studying imperceptible adversarial examples is that it lends itself naturally to an evaluation metric that provides the relative strength of different algorithms. We can look at the fraction of points for which some adversary can find a successful attack within the LP ball.

When evaluating unrestricted adversarial examples, we don’t have the same luxury. To see why, imagine some defense **D** which only makes an error on a single datapoint **x**. If an attacker wants **D** to have a very low score, they can identify that **D** makes a mistake on **x**, and then submit **x** to the contest repeatedly, making **D**’s accuracy arbitrarily low. We therefore can’t rely on the statistical performance of the model. Instead, in order to win this contest, we need a model for which no adversary can find a single **x** that breaks it.

This evaluation metric aligns with common threat modeling as well. Many situations involve a “Break Once Run Everywhere” (BORE) attack. For example if an attacker writing malware finds a single undetected exploit in a system they can use it to compromise millions of computers.

#### Should some parts of the defenses be kept secret?
**TLDR: We think nothing should be secret**

Athalye, Carlini & Wagner recommend that a realistic threat model should at least “grant knowledge of the model architecture, training algorithm, and allow query access”.

We would propose the following be shared:
model architecture (via source code)
training algorithm (via source code)
query access (via implementing an API)
model weights

The following would not be shared
exact sequence of randomness during evaluation (e.g. np.seed)

#### Are you sure that the model weights should be public?
One argument for the weights being private is that this makes it easier for the defender, and the task currently seems very hard for the defenders.

We think that we should first try making the task easier for the defender by simplifying the task, rather than relying on hidden weights. Two reasons for this are that (1) model exfiltration in the real world seems extremely likely, and (2) if model weights are private, then attackers in the contest will waste a lot of energy extracting weights from models.

A somewhat weaker argument for the weights being private is that if the attackers have to dig into the guts of the defenses, then some defenses will be favored for reasons other than their true robustness. For example:
Some defensive techniques make the gradients numerically unstable and harder to attack
Some defenders might write their code in obtuse ways that are hard to read and then fewer attackers will want to mess with it

We don’t find this argument too compelling, because if we set up the incentives for prizes correctly, then we should be able to find people who are motivated to break ugly defenses. Overall, we think that it’s likely still worthwhile to make the weights public.

#### Will the contest need lots of taskers to evaluate attack images?
We think that it’s possible that we can get away with just a few taskers doing evaluations. Four things that we are optimistic about this:

1. We charge the attackers for submitting an attack, to prevent weak attackers from wasting our time. (This doesn't limit the number of samples attackers can use to test too badly, see #2)
2. Since the defending models are public, the attacker can try an attack many times themselves until they find a vulnerability, and then they can look at it themselves to see if it seems valid. They could even set up their own cheaper taskers if they want to.
3. Only a single image is needed to permanently break a model.
4. Submitting a single image could in fact permanently break several models.
 
All this considered, for a contest with N models, we could imagine having maybe ~10*N submitted images total. If we have a similar number of models as the 2017 NIPS advex competition, this would yield on the order of ~10K total images to be labeled by each of our ensemble of taskers.

