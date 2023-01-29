from arithmetic_compressor.models.base_adaptive_model import BaseFrequencyTable
from arithmetic_compressor.util import *

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
                        for sym, prob in probability.items()}
    for sym, freq in freq.items():
      cdf[sym] = Range(prev_freq, prev_freq + freq)
      prev_freq += freq
    self.cdf_object = cdf

  def cdf(self):
    return self.cdf_object

  def probability(self):
    return self.__prob

  def predict(self, symbol):
    assert symbol in self.symbols
    return self.probability()[symbol]

  def update(self, symbol):
    pass

  def test_model(self, gen_random=True, N=10000, custom_data=None):
    self.name = "Static Model"
    return BaseFrequencyTable.test_model(self, gen_random, N, custom_data)
