from arithmetic_compressor.models.base_adaptive_model import *
from arithmetic_compressor.models.ppm import MultiPPM
from arithmetic_compressor.util import *

"""
Binary PPM

Prediction by partial matching (PPM) for binary symbols, (0 and 1)
Uses Fixed Point arithmetic to update probabilities
"""


UPDATE_RATE = 9


class BaseBinaryModel(BaseFrequencyTable):
  """A Binary Adaptive Model, similar to the SimpleAdaptiveModel,
  but specifically designed for binary symbols (0 and 1).
  """

  def __init__(self, update_rate=UPDATE_RATE):
    super().__init__({0: .5, 1: .5})

    self.name = "Base Binary"
    self.update_rate = update_rate
    self.prob_1_scaled = self.scale_factor >> 1  # P(1) = 0.5

  def update(self, symbol):
    prob_1 = self.prob_1_scaled
    if symbol == '0':
      prob_1 -= self.prob_1_scaled >> self.update_rate
    else:
      prob_1 += (self.scale_factor - self.prob_1_scaled) >> self.update_rate
    self.prob_1_scaled = prob_1

  def probability(self):
    p1 = self.prob_1_scaled / self.scale_factor
    return {0: 1 - p1, 1: p1}

  def cdf(self, context=None):
    prob_1_scaled = self.prob_1_scaled
    if isinstance(context, int):
      prob_1_scaled = context
    return {
        1: Range(0, prob_1_scaled),
        0: Range(prob_1_scaled, self.scale_factor)
    }


class BinaryPPM(BaseBinaryModel):
  """Prediction by partial matching for binary symbols, 0 and 1
  """

  def __init__(self, k=3, check_lower=True):
    super().__init__()
    assert (-1 < k)
    self.name = f"Binary-PPM<{k}>"
    self.context_size = k
    self.check_lower = check_lower
    self.context = ""
    # each object will hold the probability of a 1 for all contexts encountered
    self.prob_table = [{} for _ in range(k+1)]
    self.default_prob = self.prob_1_scaled

  def _get_context_prob(self, context=None):
    """Return scaled probability of '1' from a given context
    """

    if not context:
      single = self.prob_table[0]
      prob_1_scaled = single[''] if len(single) else self.default_prob
      return prob_1_scaled

    # get last k(context size) symbols from context
    context = context[-self.context_size:]
    for s in range(len(context), -1, -1):
      if context in self.prob_table[s]:
        prob_1_scaled = self.prob_table[s][context]
        return prob_1_scaled
      context = context[1:]
      if not self.check_lower:
        break

    return self._get_context_prob()

  def update(self, symbol: str):
    assert (symbol in self.symbols)  # 0 or 1
    if len(self.context) > self.context_size:
      self.context = self.context[-self.context_size:]

    context = self.context
    for i in range(len(context)+1):
      suffix = context[i:]
      ln = len(suffix)
      if suffix not in self.prob_table[ln]:
        self.prob_table[ln][suffix] = self.default_prob

      prob_1_scaled = self.prob_table[ln][suffix]
      if symbol == '0':
        prob_1_scaled -= prob_1_scaled >> UPDATE_RATE
      else:
        prob_1_scaled += (self.scale_factor - prob_1_scaled) >> UPDATE_RATE

      self.prob_table[ln][suffix] = prob_1_scaled
      if not self.check_lower and i == 1:
        break

    if self.context_size > 0:
      self.context += str(symbol)

  def probability(self):
    prob_1_scaled = self._get_context_prob(self.context)
    p1 = prob_1_scaled / self.scale_factor
    return {1: p1, 0: 1 - p1}

  def predict(self, symbol):
    return self.probability()[symbol]

  def cdf(self):
    prob_1_scaled = self._get_context_prob(self.context)
    return super().cdf(prob_1_scaled)


class OrderN_PPM(BinaryPPM):
  """Prediction by partial matching for binary symbols 0 and 1,
  without the need for a loop to check lower context sizes (< k).
  """

  def __init__(self, k=3):
    super().__init__(k)
    assert (-1 < k)
    self.name = f"Binary-PPM-single<{k}>"
    self.prob_table = {}  # will hold the probability of a 1 for all contexts encountered

  def _get_context_prob(self, context=None):
    """Return scaled probability of '1' from a given context
    """
    context = context[-self.context_size:]
    if context in self.prob_table:
      prob_1_scaled = self.prob_table[context]
      return prob_1_scaled
    return self.default_prob

  def update(self, symbol: str):
    assert (symbol in self.symbols)  # 0 or 1
    if len(self.context) > self.context_size:
      self.context = self.context[-self.context_size:]
    elif len(self.context) < self.context_size:
      self.context += str(symbol)
      return

    context = self.context
    if context not in self.prob_table:
      self.prob_table[context] = self.default_prob

    prob_1_scaled = self.prob_table[context]
    if str(symbol) == '0':
      prob_1_scaled -= prob_1_scaled >> UPDATE_RATE
    else:
      prob_1_scaled += (self.scale_factor - prob_1_scaled) >> UPDATE_RATE

    self.prob_table[context] = prob_1_scaled

    # add symbol to context
    if self.context_size > 0:
      self.context += str(symbol)


class MultiBinaryPPM(MultiPPM):
  """Mix multiple Binary PPM models to make prediction.
  Extends MultiPPM class which uses weighted average to combine proabilities
  """

  def __init__(self, models=6):
    assert (models >= 2)
    super().__init__([0, 1], 3)
    self.name = f"Multi-Binary-PPM<0-{models}>"
    self.models = [OrderN_PPM(k) for k in range(models+1)]
    # self.models = [BinaryPPM(k, False) for k in range(models+1)] # little bit slower
    self.weights = [1/len(self.models)] * len(self.models)
