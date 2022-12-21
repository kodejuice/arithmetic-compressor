from misc import *

RATE = 10  # the closer to 0 this is, the faster the probabilities adapts
ADAPT_RATE = 1 - 1 / (1 << RATE)


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
    self.scale_factor = 4096

    # private
    self.__freq = {sym: 0 for sym in self.symbols}
    self.__freq_total = 0
    self.__prob = dict(probability)

  def update(self, symbol, context=None):
    self.__freq[symbol] = self.__freq.get(symbol, 0) + 1
    self.__freq_total += 1
    self.__prob = {k: f/self.__freq_total for k, f in self.__freq.items()}

  def freq(self, context=None):
    return self.__freq

  def scaled_freq(self, context=None):
    P = self.probability(context).items()
    freq = {sym: round(self.scale_factor * prob) for sym, prob in P}
    return freq

  def cdf(self, context=None):
    """Create a cummulative distribution function from a probability dist.
    """
    cdf = {}
    prev_freq = 0
    freq = self.scaled_freq(context).items()
    for sym, freq in freq:
      cdf[sym] = Range(prev_freq, prev_freq + freq)
      prev_freq += freq
    return cdf

  def probability(self, context=None, return_array=False):
    if return_array:
      # for rANS
      return [self.__prob[sym] for sym in self.symbols]
    return self.__prob

  def predict(self, symbol, context=None):
    return self.probability(context)[symbol]

  def entropy(self, N=1):
    return HF(self.freq())

  def test_model(self, gen_random=True, N=10000):
    """Test efficiency of the adaptive frequency model
      to predict symbols
    """
    if gen_random:
      symbol_pool = [random.choice(self.symbols) for _ in range(N)]
    else:
      symbol_pool = generate_data(self.probability(), N, False)

    error = 0
    context = ""
    for i in range(N-1):
      p = self.probability(context)
      rand_symbol = symbol_pool[i]
      error += 1 - p[rand_symbol]
      self.update(rand_symbol, context)
      context += rand_symbol
    print(f"percentage of error({self.name}): {error/N}")


class SimpleAdaptiveModel(BaseFrequencyTable):
  def __init__(self, probability: dict, update_rate=ADAPT_RATE):
    assert (0 <= update_rate <= 1)
    super().__init__(probability)
    self.name = "Simple adaptive"

    self.__freq = {}
    self.__prob = dict(probability)
    self.update_rate = update_rate

  def update(self, symbol, context=None):
    assert (symbol in self.symbols)
    self.__freq[symbol] = self.__freq.get(symbol, 0) + 1
    for sym, prob in self.__prob.items():
      if sym == symbol:
        self.__prob[sym] = prob * self.update_rate + (1 - self.update_rate)
      else:
        self.__prob[sym] *= self.update_rate

  def freq(self, context=None):
    return self.__freq

  def probability(self, context=None, return_array=False):
    return self.__prob


class PPMModel(SimpleAdaptiveModel):
  """Prediction by partial matching model
  table -> [context-size][context][probability of symbols]
  e.g, table[1]['1'] = {'0': 0.5, '1': 0.5}
       table[2]['01'] = {'0': 0.73, '1': 0.27}
  """

  def __init__(self, symbols: dict, k=3, check_lower_models=True, update_rate=ADAPT_RATE):
    super().__init__({sym: -1 for sym in symbols}, update_rate)
    assert (-1 < k)
    self.name = f"PPM<{k}>"
    self.context_size = k
    self.check_lower_models = check_lower_models
    self.table = {k: {} for k in range(k+1)}

  def get_context_probability(self, context=None):
    """Return probability of a given context
    """
    if not context:
      prob = dict(self.table[0]['']) if len(self.table[0]) else {}
      if len(prob) != len(self.symbols):
        len_sym = len(self.symbols)
        return {sym: 1/len_sym for sym in self.symbols}
      return prob

    # get last k(context size) symbols from context
    context = context[(len(context) - self.context_size):]
    for s in range(len(context), -1, -1):
      if context in self.table[s]:
        return self.table[s][context]
      context = context[1:]

      if not self.check_lower_models:
        break

    # just get the stat with an empty context
    return self.get_context_probability()

  def update(self, symbol: str, context: str):
    # get last k(context size) symbols from context
    context = context[(len(context) - self.context_size):]
    assert (symbol in self.symbols)
    for i in range(len(context)+1):
      suffix = context[i:]
      ln = len(suffix)
      if suffix not in self.table[ln]:
        self.table[ln][suffix] = {}

      T = self.table[ln][suffix]
      N = len(self.symbols)
      if len(T) == 0:
        for sym in self.symbols:
          T[sym] = 1 / N

      # update probabilities
      for sym, prob in T.items():
        if sym == symbol:
          T[sym] = prob * self.update_rate + (1 - self.update_rate)
        else:
          T[sym] *= self.update_rate

      if not self.check_lower_models and i == 1:
        break

  def freq(self, context=None):
    return self.scaled_freq(context)

  def probability(self, context=None, return_array=False):
    prob = self.get_context_probability(context)
    if return_array:
      return [prob[sym] for sym in self.symbols]
    return prob


class MultiPPM(PPMModel):
  """Mix multiple PPM models to make prediction.
  Uses weighted average to combine proabilities
  """

  def __init__(self, symbols: dict, models=6, check_lower=False):
    super().__init__(symbols)
    assert (models > 1)
    self.name = f"Multi-PPM<0-{models}>"
    self.models = [PPMModel(symbols, k, check_lower) for k in range(models+1)]
    self.weights = [1/len(self.models)] * len(self.models)

  def update_weights(self, symbol: str, context: str):
    # Update the weights of the models based on their prediction accuracy
    weights = self.weights
    for i, model in enumerate(self.models):
      weights[i] *= model.predict(symbol, context)
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

  def probability(self, context=None, return_array=False):
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


# Binary models

class BaseBinaryModel(BaseFrequencyTable):
  """Binary Adaptive model
  """

  def __init__(self, update_rate=RATE):
    super().__init__({'0': .5, '1': .5})

    self.name = "Base Binary"
    self.update_rate = update_rate
    self.prob_1_scaled = self.scale_factor >> 1  # 0.5, range -> [31, 4065]

  def update(self, symbol, context=None):
    if symbol == '0':
      self.prob_1_scaled -= self.prob_1_scaled >> self.update_rate
    else:
      self.prob_1_scaled += (self.scale_factor -
                             self.prob_1_scaled) >> self.update_rate

  def probability(self, context=None):
    p1 = self.prob_1_scaled / self.scale_factor
    return {'0': 1 - p1, '1': p1}

  def cdf(self, context=None):
    prob_1_scaled = self.prob_1_scaled
    if isinstance(context, int):
      prob_1_scaled = context
    return {
        '1': Range(0, prob_1_scaled),
        '0': Range(prob_1_scaled, self.scale_factor)
    }


class BinaryPPM(BaseBinaryModel):
  """Prediction by partial matching for binary symbols
  0 and 1
  """

  def __init__(self, k=3, check_lower=True, update_rate=RATE):
    super().__init__()
    assert (-1 < k)
    self.name = f"Binary-PPM<{k}>"
    self.context_size = k
    self.check_lower = check_lower
    self.prob_table = {k: {} for k in range(k+1)}
    self.default_prob = self.prob_1_scaled
    self.update_rate = update_rate

  def _get_context_prob(self, context=None):
    """Return scaled probability of '1' from a given context
    """

    if not context:
      single = self.prob_table[0]
      prob_1_scaled = single[''] if len(single) else self.default_prob
      return prob_1_scaled

    # get last k(context size) symbols from context
    context = context[(len(context) - self.context_size):]
    for s in range(len(context), -1, -1):
      if context in self.prob_table[s]:
        prob_1_scaled = self.prob_table[s][context]
        return prob_1_scaled
      context = context[1:]
      if not self.check_lower:
        break

    return self._get_context_prob()

  def update(self, symbol: str, context: str):
    # get last k(context size) symbols from context
    context = context[(len(context) - self.context_size):]
    assert (symbol in self.symbols)  # 0 or 1
    for i in range(len(context)+1):
      suffix = context[i:]
      ln = len(suffix)
      if suffix not in self.prob_table[ln]:
        self.prob_table[ln][suffix] = self.default_prob

      prob_1_scaled = self.prob_table[ln][suffix]
      if symbol == '0':
        prob_1_scaled -= prob_1_scaled >> self.update_rate
      else:
        prob_1_scaled += (self.scale_factor -
                          prob_1_scaled) >> self.update_rate

      self.prob_table[ln][suffix] = prob_1_scaled
      if not self.check_lower and i == 1:
        break

  def probability(self, context=None):
    prob_1_scaled = self._get_context_prob(context)
    p1 = prob_1_scaled / self.scale_factor
    return {'1': p1, '0': 1 - p1}

  def predict(self, symbol, context=None):
    return self.probability(context)[symbol]

  def cdf(self, context=None):
    prob_1_scaled = self._get_context_prob(context)
    return super().cdf(prob_1_scaled)


class MultiBinaryPPM(MultiPPM):
  """Mix multiple Binary PPM models to make prediction.
  Uses weighted average to combine proabilities
  """

  def __init__(self, models=6, check_lower=False):
    assert (models >= 2)
    super().__init__(['0', '1'], 3, check_lower)
    self.name = f"Multi-Binary-PPM<0-{models}>"
    # self.models = [PPMModel(['0','1'], k, check_lower) for k in range(models+1)]
    self.models = [BinaryPPM(k, check_lower) for k in range(models+1)]
    self.weights = [1/len(self.models)] * len(self.models)
