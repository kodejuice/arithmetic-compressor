import math
import random


# return randon m-bit number
def mbit(m):
  return random.randint(2**(m - 1), 2**m - 1)


# partition string into grams of length n
def grams(s, n=1):
  r = []
  while s:
    r += [s[:n]]
    s = s[n:]
  return r


def H(X, N: str | list | int = 1):
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
  """Compute entropy from given string

  Args:
      S (str): _description_
  """
  N, freq = len(S), {}
  for c in S:
    freq[c] = freq.get(c, 0) + 1
  X = {symbol: count/N for symbol, count in freq.items()}
  # actual information content
  return H(X, S)


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
    s = shuffle_string(s)
  sym_len = len(list(probs.keys())[0])
  return grams(s, sym_len)


# shuffle a string
def shuffle_string(str):
  arr = [c for c in str]
  random.shuffle(arr)
  return "".join(arr)


class Range:
  def __init__(self, low, high):
    self.low, self.high = low, high

  def __repr__(self):
    return f"[{self.low}, {self.high})"
