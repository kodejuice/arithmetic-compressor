from arithmetic_compressor.models.base_adaptive_model import *

"""
PPM

Prediction by partial matching (PPM) is an adaptive statistical data compression technique based on context modeling and prediction.
PPM models use a set of previous symbols in the uncompressed symbol stream to predict the next symbol in the stream. 

https://en.wikipedia.org/wiki/Prediction_by_partial_matching
"""

U = 10
ADAPT_RATE = 1 - 1 / (1 << U)


class PPMModel(SimpleAdaptiveModel):
  """Prediction by partial matching model
  table -> [context-size][context][probability of next symbols]
    Holds the probability of the next symbols given a context

  e.g, table[1]['1'] = {'0': 0.2, '1': 0.8}
       table[2]['00'] = {'0': 0.73, '1': 0.27}
       ...
       table[k][...] = {...}
  """

  def __init__(self, symbols: dict, k=3, check_lower_models=True):
    super().__init__({sym: 0 for sym in symbols}, ADAPT_RATE)
    assert (-1 < k)
    self.name = f"PPM<{k}>"
    self.context_size = k
    self.check_lower_models = check_lower_models
    self.context = ""
    self.table = [{} for _ in range(k+1)]

  def get_context_probability(self, context=None):
    """Return probability of a given context
    """
    if not context:
      prob = dict(self.table[0]['']) if len(self.table[0]) else {}
      if len(prob) != len(self.symbols):
        len_sym = len(self.symbols)
        return {sym: 1/len_sym for sym in self.symbols}
      return prob

    context = context[-self.context_size:]
    for s in range(len(context), -1, -1):
      if context in self.table[s]:
        return self.table[s][context]
      context = context[1:]
      if not self.check_lower_models:
        break

    # just get the prob with an empty context
    return self.get_context_probability()

  def update(self, symbol: str):
    assert (symbol in self.symbols)
    if len(self.context) > self.context_size:
      self.context = self.context[-self.context_size:]

    context = self.context
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
      self._adapt(T, symbol)

      if not self.check_lower_models and i == 1:
        break

    if self.context_size > 0:
      self.context += str(symbol)

  def freq(self):
    return self.scaled_freq()

  def probability(self):
    prob = self.get_context_probability(self.context)
    return prob


class MultiPPM(PPMModel):
  """Mix multiple PPM models to make prediction.
  Uses weighted averaging to combine proabilities
  """

  def __init__(self, symbols: dict, models=6):
    super().__init__(symbols)
    assert (models > 1)
    self.name = f"Multi-PPM<0-{models}>"
    self.models = [PPMModel(symbols, k, False) for k in range(models+1)]
    self.weights = [1/len(self.models)] * len(self.models)

  def update_weights(self, symbol: str):
    # Update the weights of the models based on their prediction accuracy
    weights = self.weights
    for i, model in enumerate(self.models):
      weights[i] *= model.predict(symbol)
      weights[i] = max(weights[i], 0.00001)  # Additive smoothing
    # Normalize the weights so they sum to 1
    total_weights = sum(weights)
    weights = [weight / total_weights for weight in weights]
    self.weights = weights

  def update(self, symbol: str):
    for model in self.models:
      model.update(symbol)

    # update the weights of the models
    self.update_weights(symbol)

  def freq(self):
    # all models should have the same frequency
    return self.models[-1].freq()

  def probability(self):
    # Combine the probabilities using a weighted average
    combined_probs = {}
    for symbol in self.symbols:
      symbol_prob = 0
      for i in range(len(self.models)):
        symbol_prob += self.weights[i] * \
            self.models[i].predict(symbol)
      combined_probs[symbol] = symbol_prob
    return combined_probs
