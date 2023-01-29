from arithmetic_compressor.util import *
from collections import OrderedDict
from arithmetic_compressor.models.base_adaptive_model import BaseFrequencyTable

# Adapted from the PAQ6 compressor
# https://cs.fit.edu/~mmahoney/compression/paq6.cpp
# Minimal implementation, doesn't include contexts only models
# Only 3 submodels implemented (Default, CharModel, MatchModel)

"""
Context mixing (Linear Evidence Mixing)

Context mixing is related to prediction by partial matching (PPM) in that the compressor
is divided into a predictor and an arithmetic coder, but differs in that the next-symbol
prediction is computed using a weighted combination of probability estimates from a large
number of models conditioned on different contexts.
Unlike PPM, a context doesn't need to be contiguous.

In linear mixing a probability is expressed as a count of zeros and ones. Each model maps a set of distinct
contexts to a pair of counts, n0, a count of zero bits, and n1, a count of 1 bits.
In order to favor recent history, half of the count over 2 is discarded when the opposite
bit is observed.
For example, if the current state associated with a context is (n0, n1) = (12,3) and a 1 is observed,
then the counts are updated to (7, 4).

Probabilities are combined by weighted addition of the counts. Weights are adjusted in the
direction that minimizes coding cost in weight space.

http://mattmahoney.net/dc/dce.html#Section_43
https://en.wikipedia.org/wiki/PAQ#Algorithm
en.wikipedia.org/wiki/Context_mixing#Linear_Mixing
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

  def __init__(self, N, max_hash_size=500):
    super().__init__()
    self.N = N  # no of bytes to match
    self.window = ""
    self.max_hash_size = max_hash_size
    self.counter = Counter()
    self.hash = OrderedDict()

  def update(self, bit):
    if len(self.window) == 8*self.N:
      if len(self.hash) == self.max_hash_size:
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
    self.models = models or [
        # DefaultModel(),

        CharModel(32),

        MatchModel(1),
        MatchModel(2),
        MatchModel(4),
    ]
    weights, contexts = [], []
    for model in self.models:
      model_contexts = model.get_counts()
      contexts += [model_contexts]
      weights += [[1]*len(model_contexts)]
    self.weights = weights
    self.contexts = contexts

  def __update_weights(self, bit):
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
    self.__update_weights(bit)

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
    p1 = max(p1, 0.001)
    return {1: p1, 0: 1-p1}

  def cdf(self):
    """Create a cummulative distribution function from the probability of 0 and 1.
    """
    p = self.probability()
    p1 = round(PSCALE * p[1])
    return {1: Range(0, p1), 0: Range(p1, PSCALE)}

  def test_model(self, gen_random=True, N=10000, custom_data=None):
    self.symbols = [0, 1]
    self.name = "Context Mixing<Linear>"
    return BaseFrequencyTable.test_model(self, gen_random, N, custom_data)
