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

  def predict(self, symbol, context=None):
    return self.probability(context)[symbol]

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

  def __init__(self, symbols: dict, k=3, check_lower_models=True):
    symbols = list(symbols)
    super().__init__({k: 1/len(symbols) for k in symbols})
    assert (-1 < k)
    self.name = f"PPM<{k}>"
    self.context_size = k
    self.check_lower_models = check_lower_models
    self.table = {k: {} for k in range(k+1)}

  def _get_context_stat(self, context=None):
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

      if not self.check_lower_models:
        break

    # just get the stat with an empty context
    return self._get_context_stat()

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
      if not self.check_lower_models and i == 1:
        break

  def freq(self, context=None):
    _, freq = self._get_context_stat(context)
    return freq

  def probability(self, context: list | None = None, return_array=False):
    prob, _ = self._get_context_stat(context)
    if return_array:
      return [prob[sym] for sym in self.symbols]
    return prob


class MultiPPM(PPMModel):
  """Mix multiple PPM models to make prediction.
  Uses weighted average to combine proabilities
  """

  def __init__(self, symbols: dict, check_lower=False):
    super().__init__(symbols)
    self.name = "PPM-mix<0-5>"
    self.models = [PPMModel(symbols, k, check_lower) for k in range(6)]
    self.weights = [1/len(self.models)] * len(self.models)

  def update_weights(self, symbol: str, context: str):
    # Update the weights of the models based on their prediction accuracy
    weights = self.weights
    symbol_true_probability = 1
    # symbol_true_probability = (context+symbol).count(symbol) / (len(context)+1)
    for i, model in enumerate(self.models):
      error = abs(model.predict(symbol, context) - symbol_true_probability)
      weights[i] *= 1 - error
      weights[i] = max(weights[i], 0.00001)  # Additive smoothing
    # Normalize the weights so they sum to 1
    total_weights = sum(weights)
    weights = [weight / total_weights for weight in weights]
    self.weights = weights

  def update(self, symbol: str, context: str):
    for model in self.models:
      model.update(symbol, context)

    # update the weights of the models
    self.update_weights(symbol, context)

  def freq(self, context=None):
    # all models should have the same frequency
    return self.models[-1].freq(context)

  def probability(self, context: list | None = None, return_array=False):
    return self._combine_probabilities(context)

  def _combine_probabilities(self, context=None):
    # Combine the probabilities using a weighted average
    combined_probs = {}
    for symbol in self.symbols:
      symbol_prob = 0
      for i in range(len(self.models)):
        symbol_prob += self.weights[i] * \
            self.models[i].predict(symbol, context)
      combined_probs[symbol] = symbol_prob
    return combined_probs

