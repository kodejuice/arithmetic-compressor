#!/usr/bin/env py
from compressors import AECompressor, rANSCompressor
import random
from misc import *


class BaseFrequencyTable:
  """Base frequency table class
  """

  def __init__(self, probability: dict):
    symbols = list(probability.keys())
    if len(symbols) < 2:
      exit("Invalid symbol length: 1, add more symbols")

    self.name = "Base"
    self.symbols = symbols
    # map symbols to indexes (used in the ANS compressor)
    self._symbol_index = {symbol: i for i, symbol in enumerate(self.symbols)}

    freq = {}
    sample_input = generate_data(probability)
    for symbol in sample_input:
      freq[symbol] = freq.get(symbol, 0) + 1

    self.__freq = freq
    self.__freq_total = sum(freq.values())
    self.__prob = {k: f/self.__freq_total for k, f in self.__freq.items()}

  def update(self, symbol, context=None):
    self.__freq[symbol] += 1
    self.__freq_total += 1
    self.__prob = {k: f/self.__freq_total for k, f in self.__freq.items()}

  def freq(self, context=None):
    return self.__freq

  def cdf(self, context=None):
    """Create a cummulative distribution function from a probability dist.
    """
    cdf = {}
    prev_freq = 0
    for sym, freq in list(self.freq(context).items()):
      cdf[sym] = Range(prev_freq, prev_freq + freq)
      prev_freq += freq
    return cdf

  def probability(self, context: list | None = None, return_array=False):
    if return_array:
      # for rANS
      return [self.__prob[sym] for sym in self.symbols]
    return self.__prob

  def entropy(self, N: list | int = 1):
    """Compute entropy of prob. distribution
    Optionally return expected information content given the input data or its size

    Args:
        N (list | int, optional): input data or its size. Defaults to 1.
    """
    return H(self.probability(), N)

  def test_model(self, gen_random=True, N=10000):
    """Test efficiency of the adaptive frequency model
      to predict symbols
    """
    if gen_random:
      symbol_pool = [random.choice(self.symbols) for _ in range(N)]
    else:
      symbol_pool = generate_data(self.probability(), N, False)

    cp = 0
    context = ""
    for i in range(N-1):
      p = self.probability(context)
      rand_symbol = symbol_pool[i]
      cp += 1 - p[rand_symbol]
      self.update(rand_symbol, context)
      context += rand_symbol
    print(f"percentage of error({self.name}): {cp/N}")


class PPMModel(BaseFrequencyTable):
  """Prediction by partial matching model
  """

  def __init__(self, probability: dict, k=3):
    super().__init__(probability)
    assert (-1 < k)
    self.name = f"PPM<{k}>"
    self.context_size = k
    self.table = {k: {} for k in range(k+1)}

  def get_context_stat(self, context):
    """Return probability and frequency of a given context
    """
    if not context:
      freq = dict(self.table[0]['']) if len(self.table[0]) else {}
      if len(freq) != len(self.symbols):
        for c in self.symbols:  # add missing symbols
          freq[c] = freq.get(c, 1)
      freq_total = sum(freq.values())
      prob = {k: f/freq_total for k, f in freq.items()}
      return (prob, freq)

    # get last k(context size) symbols from context
    context = context[(len(context) - self.context_size):]
    for s in range(len(context), -1, -1):
      if context in self.table[s]:
        context_freq_table = self.table[s][context]
        # if some symbols aren't in this table,
        # initialize them with value 1
        for c in self.symbols:
          if c not in context_freq_table:
            context_freq_table[c] = 1

        # update frequency and probability table
        freq = context_freq_table
        freq_total = sum(freq.values())
        prob = {k: f / freq_total for k, f in freq.items()}

        return (prob, freq)

      context = context[1:]

    # just get the stat with an empty context
    return self.get_context_stat()

  def update(self, symbol: str, context: str):
    # get last k(context size) symbols from context
    context = context[(len(context) - self.context_size):]
    assert (symbol in self.symbols)
    for i in range(len(context)+1):
      suffix = context[i:]
      ln = len(suffix)
      if suffix not in self.table[ln]:
        self.table[ln][suffix] = {}
      if symbol not in self.table[ln][suffix]:
        self.table[ln][suffix][symbol] = 0
      self.table[ln][suffix][symbol] += 1

  def freq(self, context=None):
    _, freq = self.get_context_stat(context)
    return freq

  def probability(self, context: list | None = None, return_array=False):
    prob, _ = self.get_context_stat(context)
    if return_array:
      return [prob[sym] for sym in self.symbols]
    return prob


s = 1/7
prob = {
    '0': s,
    '1': s,
    '2': s,
    '3': s,
    '4': s,
    '5': s,
    '6': s,
}

bmodel = BaseFrequencyTable(prob)
pmodel = PPMModel(prob, 0)


AEC = AECompressor(bmodel)
# rANSC = rANSCompressor(bmodel)

# data = generate_data(prob, 100000, shuffle=True)
data = "222445613"*20000
print(len(data))
# data = "11111111110000000000111"
# data = bin(mbit(256))[2:]
# data = "1111111111010000"
# print(data, len(data))

# ANS
# print('rANS')
# ans_encoded = rANSC.compress(data)
# print(len(ans_encoded)*32, rANSC.bmodel.entropy(len(data)))
# ans_decoded = rANSC.decompress(ans_encoded, len(data))
# print(ans_decoded, data)
# print(ans_decoded == data)

# AE
print('AE')
# data = bin(mbit(100000))[2:]
# print(data, len(data))
ans_encoded = AEC.compress(data)
ans_decoded = AEC.decompress(ans_encoded, len(data))
print(len(ans_encoded),
      f"[entropy: {AEC.model.entropy(data)} | expected: {AEC.model.entropy(len(data))}]")
print(data == ans_decoded)
print(AEC.model.probability())


exit()

compressed = []
notcompressed = []

for N in range(2**16 - 1):
  bmodel = BaseFrequencyTable({'0': 0.5, '1': 0.5})
  AEC = AECompressor(bmodel)
  num = bin(N)[2:]
  num = (16-len(num))*"0" + num

  num_compressed = AEC.compress(num)
  zero_count = num.count('0')
  D = [f"{(num, 'to: '+str(len(num_compressed)), ('0: '+str(zero_count), '1: '+str(16-zero_count)), 'entropy: ' + str(AEC.model.entropy(num))+' / expected: '+str(AEC.model.entropy(len(num))))}"]
  if len(num_compressed) < len(num):
    compressed += D
  else:
    notcompressed += D


print("compressed:\n{}\n\nnot compressed:\n{}".format(
    '\n'.join(compressed), '\n'.join(notcompressed)))
