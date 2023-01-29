import math
import random
from arithmetic_compressor.util import *
from collections import OrderedDict
from arithmetic_compressor.models.base_adaptive_model import BaseFrequencyTable

# Adapted from the PAQ8 compressor
# https://cs.fit.edu/~mmahoney/compression/paq8l.zip
# Minimal implementation, doesn't include contexts only models

"""
Context mixing (Neural Network / Logistic Mixing)

Context mixing is related to prediction by partial matching (PPM) in that the compressor
is divided into a predictor and an arithmetic coder, but differs in that the next-symbol
prediction is computed using a weighted combination of probability estimates from a large
number of models conditioned on different contexts.
Unlike PPM, a context doesn't need to be contiguous.

PAQ7 introduced logistic mixing, which is now favored because it gives better compression.
Each model outputs a prediction (t_i) (instead of a pair of counts like in the linear mixing).
These predictions are averaged in the logistic domain. It is more general, since only a
probability is needed as input. This allows the use of direct context models and a more
flexible arrangement of different model types.

http://mattmahoney.net/dc/dce.html#Section_432
https://en.wikipedia.org/wiki/Context_mixing#Logistic_Mixing
https://en.wikipedia.org/wiki/PAQ#Neural-network_mixing
"""

PSCALE = 4096


def strech(x): return math.log(x / (1 - x))


def squash(x):
  x = max(-709, x)
  return 1 / (1 + (math.e**-x))

# stretch(x) = ln(x / (1 - x))
# squash(x) = 1 / (1 + e−x) (inverse of stretch).


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


class RunMap(Base):
  """Run Map. The state is (b,n) where b is the last bit seen (0 or 1) and
  n is the number of consecutive times this value was seen.  The initial
  state is (0,0).  The output is computed directly:

    t_i = (2b - 1)K log(n + 1).

  where K is ad-hoc, around 4 to 10.  When bit y_j is seen, the state
  is updated:

  (b,n) := (b,n+1) if y_j = b, else (y_j,1).
  """

  def __init__(self):
    super().__init__()
    self.K = 7  # arount 4 to 10
    self.state = (0, 0)

  def update(self, bit):
    super().update(bit)
    (b, n) = self.state
    self.state = (b, n+1) if bit == b else (bit, 1)

  def get_prediction(self):
    (b, n) = self.state
    return (2*b - 1) * self.K * math.log(n+1, 2)


class StationaryMap(Base):
  """Stationary Map.  The state is p, initially 1/2.  The output is
  t_i = stretch(p).  The state is updated at ad-hoc rate K (around 0.01):

  p := p + K(y_j - p)
  """

  def __init__(self):
    super().__init__()
    self.K = 0.01
    self.p = 1/2

  def update(self, bit):
    super().update(bit)
    self.p += self.K * (bit - self.p)

  def get_prediction(self):
    return strech(self.p)


class MatchModel(Base):
  """Match Model.  The state is (c,b), initially (0,0), where c is 1 if
  the context was previously seen, else 0, and b is the next bit in
  this context.  The prediction is:

    t_i := (2b - 1)Kc log(m + 1)

  where m is the length of the context.  The update rule is c := 1,
  b := y_j.  A match model can be implemented efficiently by storing
  input in a buffer and storing pointers into the buffer into a hash
  table indexed by context.  Then c is indicated by a hash table entry
  and b can be retrieved from the buffer.

  # slight deviation from PAQ8 (custom context sizes)
  """

  def __init__(self, context_sizes=[8, 16, 32, 64], max_hash_size=500):
    super().__init__()
    self.K = 0.01
    # context info
    self.context_sizes = sorted(context_sizes)
    # string containing all bits seen (our context)
    self.window = ""
    self.hash = OrderedDict()  # map a context to the next bit after it
    self.max_hash_size = max_hash_size
    self.max_window_size = 500

  def update(self, bit):
    super().update(bit)
    for size in self.context_sizes:
      if len(self.window) >= size:
        if len(self.hash) == self.max_hash_size:
          self.hash.popitem(last=False)
        context = self.window[-size:]
        self.hash[context] = bit

    if len(self.window) >= self.max_window_size:
      self.window = self.window[1:]

    self.window += str(bit)

  def get_prediction(self):
    m = 0
    K = self.K

    # last seen bit
    next_bit_in_context = self.data[-1] if len(self.data) else 0

    # check if we've seen the current context with our preset sizes
    for context_size in self.context_sizes:
      context = self.window[-context_size:]
      if context in self.hash:
        m = max(m, context_size)
        next_bit_in_context = int(self.hash[context])

    c = int(m > 0)
    b = next_bit_in_context

    return (2*b - 1) * K*c * math.log(m + 1, 2)


class NonstationaryMap(Base):
  """Nonstationary Map.  This is a compromise between a stationary map, which
  assumes uniform statistics, and a run map, which adapts quickly by
  discarding old statistics.  An 8 bit state represents (n0,n1,h), initially
  (0,0,0) where:

    n0 is the number of 0 bits seen "recently".
    n1 is the number of 1 bits seen "recently".
    n = n0 + n1.
    h is the full bit history for 0 <= n <= 4,
      the last bit seen (0 or 1) if 5 <= n <= 15,
      0 for n >= 16.
  """

  def __init__(self):
    super().__init__()
    self.state = (0, 0, 0)

  def update(self, bit):
    """The update rule is biased toward newer data in a way that allows
  n0 or n1, but not both, to grow large by discarding counts of the
  opposite bit.  Large counts are incremented probabilistically.
  Specifically, when y_j = 0 then the update rule is:

    n0 := n0 + 1, n < 29
          n0 + 1 with probability 2^(27-n0)/2 else n0, 29 <= n0 < 41
          n0, n = 41.
    n1 := n1, n1 <= 5
          round(8/3 lg n1), if n1 > 5

  swapping (n0,n1) when y_j = 1.
    """
    super().update(bit)

    (n0, n1, h) = self.state
    n0 = n0+1 if bit == 0 else 1
    n1 = n1+1 if bit == 1 else 1
    n = n0 + n1

    h = 0
    if n < 4:
      h += bit
    elif n <= 16:
      h = bit

    if bit == 0:
      n0, n1 = self._update_count(n0, n1, n)
    else:
      n1, n0 = self._update_count(n1, n0, n)

    self.state = (n0, n1, h)

  def _update_count(self, nx, ny, n):
    if n < 29:
      nx += 1
    elif 29 <= nx < 41:
      p = 2**((27 - nx) >> 1)
      if random.random() < p:
        nx += 1
      else:
        nx += nx
    if ny > 5:
      ny = round(8/3 * math.log(ny))
    return nx, ny

  def get_prediction(self):
    """The primaty output is t_i := stretch(sm(n0,n1,h)), where sm(.) is
  a stationary map with K = 1/256, initialized to
  sm(n0,n1,h) = (n1+(1/64))/(n+2/64).
    """
    def sm(n0, n1): return (n1+(1/64))/((n0+n1)+2/64)
    n0, n1, _ = self.state
    return strech(sm(n0, n1))


"""MIXER"""


class ContextMix_Logistic(Base):
  """Logistic/Neural network Mixer
  A neural network is used to combine models.  The
  i'th model independently outputs t_i, which is the streched probability
  of the model.

  The network computes the next bit probabilty
    p1 = squash(Σi w_i t_i), p0 = 1 - p1                        (1)

  p1 is the output prediction

  After bit y_j (0 or 1) is received, the network is trained:

  w_i := w_i + eta t_i (y_j - p1)                                (2)

  where eta is an ad-hoc learning rate, typically around 0.01, t_i is the
  i'th input, (y_j - p1) is the prediction error for the j'th input but,
  and w_i is the i'th weight.
  """

  def __init__(self, learning_rate=0.1):
    super().__init__()
    self.models = [
        RunMap(),
        StationaryMap(),
        NonstationaryMap(),
        MatchModel()
    ]
    self.weights = [0] * len(self.models)
    self.learning_rate = learning_rate

  def __update_weights(self, bit):
    """
    w_i := w_i + eta t_i (y - p1)
    p1 = squash(Σi wi t_i)
    where (y - p1) is the prediction error.
    Unlike linear mixing, weights can be negative.

    The probability computation is essentially a neural network evaluation taking stretched
    probabilities as input. Again we find the optimal weight update by taking the partial
    derivative of the coding cost with respect to the weights. The result is that the update
    for bit y (0 or 1) is simpler than back propagation (which would minimizes RMS error instead).
    """
    eta = self.learning_rate
    w = self.weights
    t = [model.get_prediction() for model in self.models]

    p1 = squash(sum([w[i]*t[i] for i in range(len(w))]))
    for i in range(len(self.models)):
      self.weights[i] = w[i] + eta * t[i] * (int(bit) - p1)

  def update(self, bit):
    # update models
    for model in self.models:
      model.update(bit)

    # update the weights of the models
    self.__update_weights(bit)

  def probability(self):
    """
    p1 = squash(Σi wi t_i)
    """
    w = self.weights
    t = [model.get_prediction() for model in self.models]

    p1 = squash(sum([w[i]*t[i] for i in range(len(w))]))
    p1 = max(p1, 0.01)
    return {1: p1, 0: 1-p1}

  def cdf(self):
    """Create a cummulative distribution function from the probability of 0 and 1.
    """
    p = self.probability()
    p1 = round(PSCALE * p[1])
    p1 = min(PSCALE-1, p1)
    return {1: Range(0, p1), 0: Range(p1, PSCALE)}

  def test_model(self, gen_random=True, N=10000, custom_data=None):
    self.symbols = [0, 1]
    self.name = "Context Mixing<Logistic>"
    return BaseFrequencyTable.test_model(self, gen_random, N, custom_data)
