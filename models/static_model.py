from util import *

SCALE_FACTOR = 4096


class StaticModel:
  """A static model, which does not adapt to input data or statistics.
  """

  def __init__(self, probability):
    symbols = list(probability.keys())

    self.name = "Static"
    self.symbols = symbols
    self.__prob = dict(probability)

    # compute cdf from given probability
    cdf = {}
    prev_freq = 0
    self.freq = freq = {sym: round(SCALE_FACTOR * prob)
                        for sym, prob in probability}
    for sym, freq in freq:
      cdf[sym] = Range(prev_freq, prev_freq + freq)
      prev_freq += freq
    self.cdf = cdf

  def cdf(self):
    return self.cdf

  def probability(self):
    return self.__prob

  def predict(self, symbol):
    assert symbol in self.symbols
    return self.probability()[symbol]

  def update(self, symbol):
    pass
