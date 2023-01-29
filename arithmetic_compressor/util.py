import math
import random


def H(X, N=1):
  """Compute entropy from two input types

  Args:
      X (_type_): Probability distribution
      N (str | list | int, optional): data or its size. Defaults to 1.
  """
  h = 0
  if not isinstance(N, int):
    # N is our data
    # sum up the information content of each symbol in our data
    for i in range(len(N)):
      h += -math.log2(X[N[i]])
    # number of bits needed to encode our data
    return h

  # compute expected number of bits per symbol
  for symbol in X:
    h += X[symbol] * -math.log2(X[symbol])

  # This is the expected number of bits for N symbols
  return h * N


def HF(freq_table: dict):
  """compute entropy from a frequency table

  Args:
      freq_table (dict): frequency table
  """
  h, N = 0, sum(freq_table.values())
  # h += sum([-math.log2(freq/N)] * freq)
  for sym, freq in freq_table.items():
    h += freq * -math.log2(freq/N)
  return h


def h(S: str):
  """Compute information content of a given string

  Args:
      S (str): _description_
  """
  N, freq = len(S), {}
  for c in S:
    freq[c] = freq.get(c, 0) + 1
  X = {symbol: count/N for symbol, count in freq.items()}
  return H(X, S)


# generate data, given count and probability of symbols
def generate_data(probs, N=10, shuffle=False):
  prob_sum = sum(probs.values())
  assert (prob_sum < 2)
  data = []
  for sym, prob in probs.items():
    data += [sym] * int(N * prob)
  if shuffle:
    random.shuffle(data)
  return data


class Range:
  def __init__(self, low, high):
    self.low, self.high = low, high

  def __repr__(self):
    return f"[{self.low}, {self.high})"


class Counter:
  """A Counter represents a pair (n0, n1) of counts of 0 and 1 bits
  in a context.

  This is used by the models to represent relative proabilities
  of contexts in the Context Mixing model
  """

  def __init__(self, c=None, get_counts=False):
    self.c = c or [0, 0]
    assert (len(self.c) == 2)
    self.get_counts = get_counts

  def __repr__(self) -> str:
    return f"{self.get_prob()}"

  def add(self, bit):
    if self.c[bit] < 255:
      self.c[bit] += 1
    # update oppsite bit
    ob = self.c[1-bit]
    if ob > 25:
      self.c[1-bit] = math.floor(math.sqrt(ob) + 6)
    elif ob > 1:
      self.c[1-bit] = ob >> 1

  def get_prob(self):
    c = self.c
    if self.get_counts:
      return c

    # compute relative probabilities (n0, n1)
    if c[1] >= c[0]:
      if c[0] == 0:
        n = (0, 4*c[1])
      else:
        n = (1, c[1]//c[0])
    else:
      if c[1] == 0:
        n = (4*c[0], 0)
      else:
        n = (c[0]//c[1], 1)
    return n
