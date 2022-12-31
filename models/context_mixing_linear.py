import math
from util import *
from models.base_adaptive_model import BaseFrequencyTable, SimpleAdaptiveModel

# Adapted from the PAQ6 compressor
# https://cs.fit.edu/~mmahoney/compression/paq6.cpp

"""
Context mixing (Linear Evidence Mixing)

Context mixing is related to prediction by partial matching (PPM) in that the compressor
is divided into a predictor and an arithmetic coder, but differs in that the next-symbol
prediction is computed using a weighted combination of probability estimates from a large
number of models conditioned on different contexts.
Unlike PPM, a context doesn't need to be contiguous.

A probability is expressed as a count of zeros and ones. Each model maps a set of distinct
contexts to a pair of counts, n0, a count of zero bits, and n1, a count of 1 bits.
In order to favor recent history, half of the count over 2 is discarded when the opposite
bit is observed.
For example, if the current state associated with a context is (n0, n1) = (12,3) and a 1 is observed,
then the counts are updated to (7, 4).

Probabilities are combined by weighted addition of the counts. Weights are adjusted in the
direction that minimizes coding cost in weight space.

https://mattmahoney.net/dc/dce.html#Section_43
https://en.wikipedia.org/wiki/PAQ#Algorithm
"""


"""
TODO:
- Improve speed of OrderN model
- Add get_counts() to BaseFrequencyTable
- Test with other models
"""

PSCALE = 4096


class Counter:
  """A Counter represents a pair (n0, n1) of counts of 0 and 1 bits
  in a context.
  """

  def __init__(self, c=None):
    self.c = c or [0, 0]
    assert(len(self.c) == 2)

  def __repr__(self) -> str:
    return f"{self.get_context_counts()}"

  def add(self, bit):
    if self.c[bit] < 255:
      self.c[bit] += 1
    # update oppsite bit
    ob = self.c[1-bit]
    if ob > 25:
      self.c[1-bit] = math.floor(math.sqrt(ob) + 6)
    elif ob > 1:
      self.c[1-bit] = ob >> 1

  def get_prob(self):
    c = self.c
    # compute relative probabilities (n0, n1)
    if c[1] >= c[0]:
      if c[0] == 0:
        n = (0, 4*c[1])
      else:
        n = (1, c[1]//c[0])
    else:
      if c[1] == 0:
        n = (4*c[0], 0)
      else:
        n = (c[0]//c[1], 1)
    return n


class Base:
  def __init__(self):
    self.data = []
    self.current_byte = []

  def update(self, bit):
    bit = int(bit)
    assert bit == 0 or bit == 1
    if len(self.current_byte) == 8:
      self.data += self.current_byte
      self.current_byte = []
    self.current_byte += [bit]


"""SUBMODELS"""


class DefaultModel(Base):
  """DefaultModel predicts P(1) = 0.5
  """

  def __init__(self):
    super().__init__()
    self.n = (1, 1)

  def get_counts(self):
    return [self.n]


class CharModel(Base):
  """A CharModel contains n-gram models from 0 to N.
    A context consists of the last 0 to N whole
    bytes, plus the 0 to 7 bits of the partially read current byte
  """

  def __init__(self, N=8):
    super().__init__()
    self.N = N
    self.counters = [Counter() for _ in range(N)]

  def get_counts(self):
    return self.counters

  def update(self, bit):
    bit = int(bit)
    for cp in self.counters:
      cp.add(bit)

    super().update(bit)

    if len(self.current_byte) == 1:  # start of a new byte
      for i in range(self.N-1, 0, -1):
        self.counters[i].c = self.counters[i-1].c
      self.counters[0].c = [0, 1] if bit else [1, 0]


# m = CharModel(10)
# # D = "10101010010101001010101010010101010101010010101010001001010111111111111100101111"
# D = generate_data({'0': 0.5, '1': 0.5}, 8*50+3, True)
# print(D[-8:])
# for c in D:
#   m.update(c)
# print('Char  ', m.get_counts())


class MatchModel(DefaultModel):
  pass


"""MIXER"""


class ContextMix_Linear(Base):
  """Linear Mixer
  The mixer computes a probability by a weighted summation of the N models.
  Each model outputs two numbers, n0 and n1 represeting the relative probability
  of a 0 or 1, respectively. These are combined using weighted summations to
  estimate the probability p that the next bit will be a 1:

        SUM_i=1..N w_i n1_i
    p = -------------------,  n_i = n0_i + n1_i
        SUM_i=1..N w_i n_i

  The weights w_i are adjusted after each bit of uncompressed data becomes
  known in order to reduce the cost (code length) of that bit.  The cost
  of a 1 bit is -log(p), and the cost of a 0 is -log(1-p).  We find the
  gradient of the weight space by taking the partial derivatives of the
  cost with respect to w_i, then adjusting w_i in the direction
  of the gradient to reduce the cost.
  """

  def __init__(self, models=None) -> None:
    super().__init__()
    self.custom_models = models != None
    self.models = models or [
        CharModel(16)
    ]
    weights, contexts = [], []
    for model in self.models:
      model_contexts = model.get_counts()
      contexts += [model_contexts]
      weights += [[1]*len(model_contexts)]
    self.weights = weights
    self.contexts = contexts

  def update_weights(self, bit):
    """
    wi ← wi + [(S n1i - S1 ni) / (S0 S1)] error.
    """
    s0, s1 = 0, 0
    for c in range(len(self.models)):
      for j in range(len(self.contexts[c])):
        context_weight = self.weights[c][j]
        n0i, n1i = self.contexts[c][j].get_prob()
        s0 += context_weight * n0i
        s1 += context_weight * n1i

    if s0 > 0 and s1 > 0:
      S = s0+s1
      P1 = s1 / S
      for c in range(len(self.models)):
        for j in range(len(self.contexts[c])):
          n0i, n1i = self.contexts[c][j].get_prob()
          ni = n0i + n1i
          error = int(bit) - P1
          self.weights[c][j] += (((S*n1i) - (s1*ni)) / (s0 * s1)) * error

  def update(self, bit, context=None):
    # update models
    for model in self.models:
      model.update(bit)

    # update the weights of the models
    self.update_weights(bit)

  def probability(self, context=None):
    """
    Linear Evidence Mixing
    Let n0i and n1i be the counts of 0 and 1 bits for the i'th model.
    The combined probabilities p0 and p1 that the next bit will be a 0 or 1 respectively,
    are computed as follows:

    s0 = Σi wi·n0i = evidence for 0
    s1 = Σi wi·n1i = evidence for 1
    S = s0 + s1 = total evidence
    p0 = s0/S = probability that next bit is 0
    p1 = s1/S = probability that next bit is 1
    """
    s0, s1 = 1, 1
    for c in range(len(self.models)):
      for j in range(len(self.contexts[c])):
        context_weight = self.weights[c][j]
        n0i, n1i = self.contexts[c][j].get_prob()
        s0 += context_weight * n0i
        s1 += context_weight * n1i
    sum = s0 + s1
    p1 = s1 / sum
    return {'1': p1, '0': 1-p1}

  def cdf(self):
    """Create a cummulative distribution function from the probability of 0 and 1.
    """
    p = self.probability()
    p0 = round(PSCALE * p['0'])
    return {
        '0': Range(0, p0),
        '1': Range(p0, PSCALE)
    }

  def entropy(self):
    m = self.models[-1]  # all models should have same data
    return h(m.data + m.current_byte)

  def test_model(self, gen_random=True, N=10000, custom_data=None):
    """Test efficiency of the adaptive model
      to predict symbols
    """
    self.symbols = ['0', '1']
    self.name = "Context Mixing<Linear>"
    BaseFrequencyTable.test_model(self, gen_random, N, custom_data)

# ε
