import math
import random
from random import randint as R


def rand_arr(n, a, b):
  return [R(a, b) for i in range(n)]


# return randon m bit number
def mbit(m):
  return R(2**(m - 1), 2**m - 1)


# entropy
def H(X, N: list | int = 1):
  h = 0
  if not isinstance(N, int):
    # N is our data
    # sum up the information content of each symbol in our data
    for i in range(len(N)):
      h += -math.log2(X[N[i]])
    # number of bits needed to encode our data
    return h

  # expected number of bits per symbol
  for symbol in X:
    h += X[symbol] * -math.log2(X[symbol])

  # if N > 1
  #  then this is the expected number of bits for N symbols
  return h * N


# generate string, given count and probability
def generate_data(probs, N=10, shuffle=False):
  assert (sum(probs.values()) <= 1)
  s = ""
  maxProb, maxS = None, None
  for sym, prob in probs.items():
    if not maxProb:
      maxProb, maxS = prob, sym
    elif prob > maxProb:
      maxProb, maxS = prob, sym
    s += sym * int(N * prob)
  if len(s) < N:
    r = N - len(s)
    s += maxS * r
  if shuffle:
    s = list(s)
    random.shuffle(s)
    s = "".join(s)
  return s


class Range:
  def __init__(self, low, high):
    self.low, self.high = low, high

  def __repr__(self):
    return f"[{self.low}, {self.high})"

# class PPMModel:
#   def __init__(self):
#     self.contexts = {}  # A dictionary to store the contexts and their statistics

#   def update(self, symbol):
#     # Update the statistics for each context that includes the symbol
#     for context in self.contexts:
#       context_and_symbol = context + symbol
#       self.contexts[context_and_symbol] = self.contexts.get(context_and_symbol, 0) + 1
#     self.contexts[symbol] = self.contexts.get(symbol, 0) + 1

#   def predict_symbol_probability(self, context):
#     # Look up the statistics for the given context
#     context_statistics = self.contexts.get(context, None)
#     if context_statistics is None:
#       return 0.0  # No statistics available for the given context

#     # Calculate the probability of the next symbol based on the context statistics
#     total_count = sum(context_statistics.values())
#     symbol_count = context_statistics.get(symbol, 0)
#     symbol_probability = symbol_count / total_count

#     return symbol_probability

# def create_initial_model():
#   model = PPMModel()
#   for i in range(256):
#     symbol = chr(i)
#     model.update(symbol)
#   return model

# def compress_with_ppm_and_arithmetic_coding(data, context_length):
#   # Create the initial statistical model
#   model = create_initial_model()

#   # Initialize the arithmetic coder
#   coder = create_arithmetic_coder()

#   # Use the model to compress the data
#   for i in range(len(data)):
#     context = data[i-context_length:i]  # The context is the previous `context_length` symbols
#     symbol = data[i]  # The symbol to be predicted and encoded

#     # Use the model to predict the probability of the next symbol
#     symbol_probability = model.predict_symbol_probability(context)

#     # Encode the symbol using the arithmetic coder and the predicted probability
#     coder.encode_symbol(symbol, symbol_probability)

#     # Update the model with the new symbol
#     model.update(symbol)

#   # Finish the arithmetic encoding and return the compressed data
#   return coder.finish()


# # improve PPM


# """
# 1. Use a larger context size: In the example above, the context size is limited to the 100 previous symbols.
# Using a larger context size can improve the accuracy of the statistical model and lead to better compression.

# 2. Use adaptive context sizes: Instead of using a fixed context size,
# the algorithm can adapt the context size based on the complexity of the input data.
# For example, in complex data with many unique symbols, a larger context size may be more effective, while in simple data with fewer unique symbols, a smaller context size may be more effective.

# 3. Use interpolation to combine multiple models: Instead of using a single statistical model,
# the algorithm can use multiple models with different context sizes and interpolate the probabilities predicted
# by each model to improve the accuracy of thepredictions.

# 4. Use escape codes to handle rare symbols: In the example above, the algorithm simply adds any new symbols it encounters to
# the model. However, this can lead to an inefficient use of memory if the input data contains many rare symbols.
# To handle this, the algorithm can use escape codes to represent groups of rare symbols, which can reduce the size of the
# model and improve the efficiency of the algorithm.
# """

# # interpolation
# # Create a list of statistical models with different context sizes
# models = [model1, model2, model3]

# # Set the interpolation weights for each model
# weights = [0.3, 0.4, 0.3]

# # Loop through the input data and make predictions
# for i in range(1, len(data)):
#     # Get the context of the previous symbols
#     context = data[max(0, i-100):i]

#     # Initialize the probability of the next symbol to 0
#     prob = 0

#     # Loop through the models and interpolate their predictions
#     for j in range(len(models)):
#         # Get the probability predicted by the model for the context
#         model_prob = models[j].get_probability(context)

#         # Interpolate the probability using the model's weight
#         prob += weights[j] * model_prob

#     # Use the interpolated probability to predict the next symbol
#     next_symbol = predict_symbol(prob)

# def bw_transform(s):
#     n = len(s)
#     m = sorted([s[i:n] + s[0:i] for i in range(n)])
#     I = m.index(s)
#     L = ''.join([q[-1] for q in m])
#     return (I, L)

# def bw_restore(I, L):
#     n = len(L)
#     X = sorted([(i, x) for i, x in enumerate(L)], key=itemgetter(1))
#     T = [None for i in range(n)]
#     for i, y in enumerate(X):
#         j, _ = y
#         T[j] = i
#     Tx = [I]
#     for i in range(1, n):
#         Tx.append(T[Tx[i - 1]])
#     S = [L[i] for i in Tx]
#     S.reverse()
#     return ''.join(S)

