import math
from collections import OrderedDict
from util import *
from models.base_adaptive_model import BaseFrequencyTable

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
- Implement SSE
- Improve PPM speed
- Start working on block mapping
"""

PSCALE = 4096


class Base:
  def __init__(self):
    self.data = []
    self.current_byte_len = 0

  def update(self, bit):
    bit = int(bit)
    assert bit == 0 or bit == 1
    if self.current_byte_len == 8:
      self.current_byte_len = 0
    self.data += [bit]
    self.current_byte_len += 1


"""SUBMODELS"""


class DefaultModel(Base):
  """DefaultModel predicts P(1) = 0.5
  """

  def __init__(self):
    super().__init__()
    self.n = (1, 1)

  def get_counts(self):
    return [Counter(self.n)]


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

    if self.current_byte_len == 1:  # start of a new byte
      for i in range(self.N-1, 0, -1):
        self.counters[i].c = self.counters[i-1].c
      self.counters[0].c = [0, 1] if bit else [1, 0]


class MatchModel(Base):
  """A MatchModel looks for a match of length n >= 8 bytes between
  the current context and previous input, and predicts the next bit
  in the previous context with weight n. the output is (n0,n1) = (w,0) or (0,w)
  (depending on the next bit) with a weight of w = length^2 / 4 (maximum 511),
  depending on the length of the context in bytes. 
  """

  def __init__(self, N, limit=500):
    super().__init__()
    self.N = N
    self.hash = OrderedDict()
    self.limit = limit
    self.window = ""
    self.counter = Counter()

  def update(self, bit):
    if len(self.window) == 8*self.N:
      if len(self.hash) == self.limit:
        self.hash.popitem(last=False)
      self.hash[self.window] = len(self.data) - 1
      self.window = self.window[1:]
    super().update(bit)
    self.window += str(bit)

    if self.current_byte_len == 1:  # Start of new byte
      if self.window in self.hash:
        end = self.hash[self.window]
        p = len(self.data) - 1
        begin = end
        while begin > 0 and p > 0 and begin != p+1 and self.data[begin] == self.data[p]:
          begin -= 1
          p -= 1

        wt = end - begin
        wt = min(wt*wt/4, 511)

        # Predict the bit found in the matching contexts
        y = int(self.data[end + 1])
        if y:
          self.counter.c = [0, wt]
        else:
          self.counter.c = [wt, 0]
        return
    self.counter.c = [0, 0]

  def get_counts(self):
    return [self.counter]


# D = "000010100000000101100101110111111000100010001001110100000000100111101101011001111110011111011011011000000101000111101100101111010000100001001100111101100011011101011110111100110000011100110000011111100111101010011111110011100010010110000001101000000001110011111001110010110011011111101011000000011001101010110110110101110100010100100010111111111101001011011111011110000111111010011011010000110010000100011000010101001001001111111101010101000000100000100110010010000001100000011010111011001011110101000010010100100010111011101111000100101100000011011101111111011001001001000010101110111101110001000000110001100010010101010000111010101111110001110011000000001111101100001111111011111110011000010110111111110000110000000100110100001000100010000010110000101001111111110011101010010100000001111011111111001101010011111101010011000101000011110101110011110110001110000000000000100101010111111000010111111100111001110111010000100011100000111101100111111110101011111111011011011111011110010001100010001010010001100000111011100110101111010001110111111111111111110011111011111110011001100000100001100010101001100000000011101001000011110101100011110000010100001110010010101000010001000100111000000000100000010101101001100000000110011011101111011010001000001100101110001011111101101111101011011110110011110110101110100111110100111101111100111000011010001000011110110111011101011000001000101110100101110111000100011000010101111111001001110010001101100000"
# M = MatchModel(1)
# for c in D:
#   M.update(c)

# print(M.hash)

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
      S = s0 + s1
      P1 = s1 / S
      for c in range(len(self.models)):
        for j in range(len(self.contexts[c])):
          n0i, n1i = self.contexts[c][j].get_prob()
          ni = n0i + n1i
          error = int(bit) - P1
          self.weights[c][j] += abs((((S*n1i) - (s1*ni)) / (s0 * s1)) * error)

  def update(self, bit):
    # update models
    for model in self.models:
      model.update(bit)

    # update the weights of the models
    self.update_weights(bit)

  def probability(self):
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
    p1 = round(PSCALE * p['1'])
    return {'1': Range(0, p1), '0': Range(p1, PSCALE)}

  def entropy(self):
    m = self.models[-1]  # all models should have same data
    return h(m.data)

  def test_model(self, gen_random=True, N=10000, custom_data=None):
    """Test efficiency of the adaptive model
      to predict symbols
    """
    self.symbols = ['0', '1']
    self.name = "Context Mixing<Linear>"
    BaseFrequencyTable.test_model(self, gen_random, N, custom_data)
