import random
from arithmetic_compressor.util import *


U = 10
ADAPT_RATE = 1 - 1 / (1 << U)


class BaseFrequencyTable:
  """Base frequency table class
  Increases frequency count and updates probability whenever
  we come across a symbol.
  """

  def __init__(self, probability: dict):
    symbols = list(probability.keys())
    if len(symbols) < 2:
      exit("Invalid symbol length: 1, add more symbols")

    self.name = "Base Frequency Table"
    self.symbols = symbols
    self.scale_factor = 4096

    # private
    self.__freq = {sym: 1 for sym in self.symbols}
    self.__freq_total = 0
    self.__prob = dict(probability)

  def update(self, symbol):
    self.__freq[symbol] = self.__freq.get(symbol, 0) + 1
    self.__freq_total += 1
    self.__prob = {k: f/self.__freq_total for k, f in self.__freq.items()}

  def freq(self):
    return self.__freq

  def scaled_freq(self):
    P = self.probability().items()
    freq = {sym: round(self.scale_factor * prob) for sym, prob in P}
    return freq

  def cdf(self):
    """Create a cummulative distribution function from a frequency dist.
    """
    cdf = {}
    prev_freq = 0
    freq = self.scaled_freq().items()
    for sym, freq in freq:
      cdf[sym] = Range(prev_freq, prev_freq + freq)
      prev_freq += freq
    return cdf

  def probability(self):
    return self.__prob

  def predict(self, symbol):
    return self.probability()[symbol]

  def entropy(self):
    return HF(self.freq())

  def test_model(self, gen_random=True, N=10000, custom_data=None):
    """Tests efficiency of the adaptive model to predict symbols
    """
    if custom_data:
      symbol_pool = list(custom_data)
    elif gen_random:
      symbol_pool = [random.choice(self.symbols) for _ in range(N)]
    else:
      symbol_pool = generate_data(self.probability(), N, False)

    error = 0
    N = len(symbol_pool)
    for i in range(N):
      p = self.probability()
      symbol = symbol_pool[i]
      error += 1 - p[symbol]
      self.update(symbol)
    # print(f"{self.name} [% error = {error/N}]")
    return (self.name, error/N, f"{self.name} [% error = {error/N}]")

  def get_counts(self):
    """Called by the Context Mixing algorithm when
    this class or any of its children are included as
    models to the context mixing model
    """
    # context mixing only allows binary symbols
    assert set(self.symbols) == set([0, 1])
    freq = self.scaled_freq()
    return [Counter([freq[0], freq[1]], True)]


class SimpleAdaptiveModel(BaseFrequencyTable):
  """ A better approach to handle changing data statistics is to gradually "forget"
  old statistics, resulting in models that respond quickly to changed input
  characteristics, making it more efficient in practice.
  The canonical "leaking" adaptive binary model is an exponential moving average.
  """

  def __init__(self, probability: dict, update_rate=ADAPT_RATE):
    assert (0 <= update_rate <= 1)
    super().__init__(probability)
    self.name = "Simple adaptive"

    self.__prob = dict(probability)
    self.__freq = {}
    self.update_rate = update_rate

  def _adapt(self, prob_object, symbol):
    '''Exponential moving average'''
    for sym, prob in prob_object.items():
      if sym == symbol:
        prob_object[sym] = prob * self.update_rate + (1 - self.update_rate)
      else:
        prob_object[sym] *= self.update_rate
      prob_object[sym] = max(prob_object[sym], 1/self.scale_factor)

  def update(self, symbol):
    assert (symbol in self.symbols)
    self.__freq[symbol] = self.__freq.get(symbol, 0) + 1
    self._adapt(self.__prob, symbol)

  def freq(self):
    return self.__freq

  def probability(self):
    return self.__prob
