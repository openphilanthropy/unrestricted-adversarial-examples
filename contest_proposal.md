# Unrestricted Adversarial Examples Contest Proposal

This is a **living contest proposal** for the unrestricted advex contest.
We expect some specifics below to change as we refine the exact mechanics of the contest, 
and we are actively seeking feedback on all details. 
We are especially interested in any corner-cases that might allow 
for attackers or defenders to “cheat” and win prizes without actually 
following the spirit of the contest outlined previously.

### Motivation

All known machine learning systems make confident and blatant errors when 
in the presence of an adversary.
We propose an ongoing, two-player contest for evaluating the safety and robustness
of machine learning systems, with a large prize pool.

We see four main advantages that this contest provides:

  1. A well-funded contest lets the world know that this problem is conspicuously open, and could get more researchers to see this as an important area.
  2. An active, ongoing contest will decrease the cycle-time between “promising idea for a defense” and “consensus that it works or doesn’t work”.
  3. Submitted attacks and defenses will be required to be open source and to conform to a standard API, lowering the barrier to entry for new researchers.
  4. The research community can monitor the current state of attacks vs defenses, and watch progress (or note the lack thereof).



### Unambiguous two-class bird-or-bicycle dataset
This contest introduces a new image dataset. 
We ask models to answer the question
"Is this an unambiguous picture of a bird, a bicycle,
or is it (ambiguous / not obvious)?".
We call this the
"bird-or-bicycle" task.

We collect our images from the [OpenImages](https://storage.googleapis.com/openimages/web/index.html)
dataset.
To obtain the ground truth label
for an image, we ask multiple human taskers to label the
image as either a **bird**, a **bicycle**, or as being
 **ambiguous / not obvious**.
Only when all taskers unanimously agree the image is obviously either a bird or
bicycle is the
image included in the dataset.
(See the [tasker instructions section](#user-content-tasker-instructions) for more details)


### Defenders
Defenders build models that output one of three labels: (a) bird, (b) bicycle, and (c) abstain.
Note that in the contest, researchers can choose to design any mechanism for abstaining
that they desire (unlike in the warm-up where we require that the defending model returns
two logits).

The objective of this contest is for defenders to build models that will never make
*confident mistakes*.
By this, we mean that given any arbitrary input
that is unambiguous either a bird or a bicycle, the model must either
classify it correctly, or must choose to abstain.
To prevent models from vacuously abstaining on every input, the model may only abstain
on 20% of an eligibility set (described below), but may abstain on any other input. 

Defenders should upload clear training code, inference code, and the trained
model to the contest organizers by following instructions developed by the community.


### Attackers
Attackers are given complete access to defenses (including the inference source code, 
training source code, and the pre-trained models).
Given this access, attackers attempt to construct 
unambiguous adversarial inputs that cause the
defense to make a *confident mistake*.
There are two requirements here: 

  1. the input must be unambiguously either a
  bird or a bicycle, [as decided by an ensemble of taskers](#user-content-tasker-instructions), and
  2. the model must assign the incorrect label (and not choose to abstain).

If both (a) all the human taskers agree the image is either an unambiguously
either a bird or bicycle,
and (b) the classifier assigns it the other label, then the attacker
wins.

The adversary will pass these adversarial images to the contest by uploading them to a
service run by the organizers.

# <a name="prizes"></a>Contest Prize Distribution Scheme

To incentivize actors to participate in this contest as both attackers and
defenders, we will provide a prize pool.

### Defender prize
Half of the prize pool will be allocated for defenders.

The first defense that goes unbroken after
having been *staked* (defined below)
for 90 days will win half of the total prize pool 
(the exact amount will be decided in the future).
We may allocate smaller prizes for
defenses that remain unbroken for smaller numbers of days.

Any researchers associated with the contest can submit defenses
but are ineligible for prizes.

### Attacker prizes

We will allocate the other half of the attack prize for the attackers.

Attackers can submit adversarial instances to staked defenses at any time.
At the end of each week (11:59:59 PM on Friday) we will collect all adversarial
examples submitted that week and publish them publicly.
We will then send the images to taskers to confirm they are 
valid (i.e., are unambiguously a picture of a bird or a bicycle).
If attacks are found to be valid, then the defenders are notified
and can appeal the image if they disagree with the taskers.
A small monetary prize will be awarded to any team that breaks a
previously-unbroken defense with an eligible input.
If a defense is broken by
multiple attacks from the same week, then the prize is split among the
teams that beat the defense.

Because we need human evaluation of attacks, we allow up to 10 attacks per
day from any team with an email that is associated with a published
arXiv paper.
Researchers interested in participating who have not submitted an arXiv
paper previously can email the review board.
Any researchers associated with the contest can
submit attacks but are ineligible for prizes.

### Eligible defenses
We want to avoid having people submitting defenses that are not novel
or real contenders to win the contest.

New defense submissions must do the following in order to be eligible:

  1. Obtain perfect accuracy on 80% of the eligibility dataset;
  2 Successfully defend against all previously-submitted adversarial examples;
  3. Run in a docker container and conform to the API maintaining a throughput
    of at least 1 image per minute per P100 GPU;
  4. Be written in readable TensorFlow, PyTorch, Caffe, or pure NumPy;
    obfuscated or exceptionally hard-to-reproduce defenses are
    ineligible (as judged by a review board).


Anyone can submit a defense at any time, and if it is eligible, we
will publish it by the end of the following week. Newly published defenses will initially
be unstaked, with no prize (other than honor) for breaking them.

### Staking an eligible defense
A *staked defense* has a small prize for attackers associated with it, and is also 
eligible to win the large defender prize if it remains unbroken for 90 days.

Defenses can become staked through one of two ways:

#### 1. A review board stakes the top defenses each week.

Every few weeks, a review board chooses the best
defenses that have been submitted, and stakes them with a small monetary prize.
Due to financial constraints we do not expect to be able to stake all defenses.

The criteria the review board will use for selecting defenses to stake are as follows


  * The defense is clearly reproducible and easy to run;
  * The review board believes that it has a non-zero chance of claiming the Defender prize
(i.e., the proposal has not been broken previously);


There is some flexibility here. For example, 
preference will be given to defenses that have been accepted to peer-reviewed venues 
and the review board may stake a high-profile defense that has been accepted at a top
conference even if it is hard to run.
Conversely, the review board expects to stake defenses that are easy to run 
before they have been accepted at any conference.)

All the attacks that target staked defenses are released together.
If several defenses that were staked in the same batch remain unbroken for the
time period specified necessary to win the defense prize, it will
be split among them.

#### 2. Any team may pay to stake their own defense themselves.
The amount paid will be equal to the monetary prize that covers the prize
that will be awarded to the attacker who breaks the defense.
If a team pays to have their defense staked, and the defense is never broken, they
will receive their initial payment back along with their portion of the defender prize.

### Review Board

We will form a *review board* of researchers who helped organize the contest.
In exceptional circumstances, the review board has the power to withhold prizes (e.g.,
in cases of cheating)
No member of the review board will be eligible to win any prize.
Members of the review board will recuse themselves when discussing submissions from
researchers they conflict with (e.g., co-authored a recent paper, conflicted
institutionally, advisor/advisee, etc).

### Appeals
Defenders who find images ambiguous that are labeled by taskers as unambiguously one class
can be appealed to the review board.
The review board will either have additional taskers label the image, or make an executive
decision.
In exceptional cases, an attacker can also appeal image labeled as ambiguous; however,
we expect it will be easier in most cases to re-upload a similar adversarial image to be
re-processed by taskers.


# Uninteresting defenses that solve the warm-up
Because the warm-up consists only of fixed attacks, we expect there will
be a wide range of defenses that easily defeat the specific fixed attacks
we selected, but don't actually solve the problem of adversarial examples
under the restircted perturbation budgets.
The motivates *why} we the full challenge is a two-player situation.


In order to demonstrate that simple uninteresting defenses exist, we have
constructed one defense that completely stops both of the supplied epsilon-ball
attacks.
(Observe that it is not possible to cheat on the rotations and translations adversarial
attack because we generate examples through brute force.)

Our "defense" is as follows:

    def np_model(x):
      x = x + np.random.normal(0, .05, size=x.shape)
      logits = sess.run(logits, {x_input: x})
      return np.array(logits == np.max(logits, axis=1), dtype=np.float32)


This defense has two pieces: (1) randomness, which breaks the decision-only
attack, and (2) gradient masking, which breaks SPSA.
Both of these facts are well-known failure modes for these types of attacks,
we describe why each is "effective" below.

Randomness prevents the decision-only attack from working because it
constantly shifts the location of the decision boundary very slightly.
For an attack that works by walking along the decision boundary,
if we move where exactly the boundary is from one run to the next, we
won't be able to move along it.

SPSA works by numerically estimating the gradient of the input image
with respect to the probabilities.
By clipping the values to be identically either `1` or `0`, we prevent
any possible gradient signal from being revealed.


# Reviewing inputs with taskers
To review inputs and classify them as either a bird, a bicycle, or
ambiguous, we utilize an ensemble of several *taskers*: humans
that we have selected to review inputs manually.
We will ask them to confirm that a given image definitely contains one
class, definitely does not contain the other, the object is not
truncated or occluded, and is a real object and not a painting, sculpture,
or other depiction of the object.

We will ask at least three taskers for each submitted image.
If any tasker is not certain, the image is rejected.
We will provide the taskers with multiple examples and continuously
monitor their responses.

Taskers will given unique IDs and all images will be released along with
all tasker's ID and response.

The review board reserves the right to over-rule the taskers, but expects
to do so only in exceptional circumstances.
If this is done, the review board will publicly explain why the result
was over-ruled.

Before a defense can win the defense prize, the review board will examine
every submitted adversarial example and confirm that all images rejected 
by taskers are in fact invalid.



# <a name="tasker-instructions"></a>Instructions given to taskers
### We provide the following instructions to taskers.

Answer the following questions:

**1. Does this photo contain a bird, or a depiction of a bird (e.g., a
toy bird, a painting of a bird, a stuffed animal bird, a cartoon bird) anywhere
in the image?**

  1. Definitely yes (and I am confident that no other tasker will guess "no")
  2. I'm not sure, but my best guess is yes
  3. I'm not sure, but my best guess is no
  4. Definitely no (and I am confident that no other tasker will guess "yes")


**2. Does this photo contain a bicycle, or a depiction of a bicycle (e.g.,
a drawing of a bicycle, a model bicycle, a toy bicycle)
anywhere in the image?**

  1. Definitely yes (and I am confident that no other tasker will guess "no")
  2. I'm not sure, but my best guess is yes
  3. I'm not sure, but my best guess is no
  4. Definitely no (and I am confident that no other tasker will guess "yes")

*If the tasker said that there definitely IS NOT one class, and
  there MAYBE IS the other class, then move on to the following
  additional questions.*

**3. For the largest single bird/bicycle in the image, 
label the pixels of the bird/bicycle. 
(If there are multiple that are the same size, then choose one at random)**

**4. Please answer the following True/False statements about the labeled bird/bicycle**

  1. This bird/bicycle is complete and not truncated. It does not go
  outside of the image at all
  2. This bird/bicycle is not occluded by anything else. I can see all of the bird/bicycle
  3. This is a picture of a **[real, live bird]** / **[real bicycle]**. It is not a painting, drawing,
  sculpture, toy, stuffed animal, or any other sort of depiction.
(It is okay if the object is a photorealistic rendering of a bird/bicycle.)


### Criteria for a valid unambiguous bird-or-bicycle image
The image is determined to be valid only if all of the following hold true:

1. all taskers answered "Definitely yes" to one class, "Definitely no" to the other class
2. the largest object is at least 25% of the image (as per the smallest bounding box by any tasker)
3. the image is not truncated, is not occluded, and is not a depiction of any sort.


# Additional Contest Mechanics FAQ

#### Why do we need a model that *never* makes a mistake? Why can’t we give partial credit to defenders in the contest?
One advantage of studying imperceptible adversarial examples is that it lends itself naturally to an evaluation metric that provides the relative strength of different algorithms. We can look at the fraction of points for which some adversary can find a successful attack within the LP ball.

When evaluating unrestricted adversarial examples, we don’t have the same luxury. To see why, imagine some defense **D** which only makes an error on a single datapoint **x**. If an attacker wants **D** to have a very low score, they can identify that **D** makes a mistake on **x**, and then submit **x** to the contest repeatedly, making **D**’s accuracy arbitrarily low. We therefore can’t rely on the statistical performance of the model. Instead, in order to win this contest, we need a model for which no adversary can find a single **x** that breaks it.

This evaluation metric aligns with common threat modeling as well. Many situations involve a “Break Once Run Everywhere” (BORE) attack. For example if an attacker writing malware finds a single undetected exploit in a system they can use it to compromise millions of computers.

#### Should some parts of the defenses be kept secret?
Athalye, Carlini & Wagner recommend that a realistic threat model should at least “grant knowledge of the model architecture, training algorithm, and allow query access”.

We therefore propose the following be shared:
  * model architecture (via source code)
  * training algorithm (via source code)
  * query access (via implementing an API)
  * model weights

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

  1. We limit the number of attacks per attacker, to prevent weak attackers from wasting our time.
  2. Since the defending models are public, the attacker can try an attack many times themselves until they find a vulnerability, and then they can look at it themselves to see if it seems valid. They could even set up their own cheaper taskers if they want to.
  3. Only a single image is needed to permanently break a model.
  4. Submitting a single image could in fact permanently break several models.
 
All this considered, for a contest with N models, we could imagine having maybe ~10*N submitted images total. If we have a similar number of models as the 2017 NIPS advex competition, this would yield on the order of ~10K total images to be labeled by each of our ensemble of taskers.


#### What are the eligibility rules for the contest? In particular, are family members or spouses of Google employees eligible?

When we figure out any prizes for the full contest, the contest organizers definitely won't be eligible for prizes. We believe (but can't commit to saying) anyone else, including google employees, will be able to get prizes.
