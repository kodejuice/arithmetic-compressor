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
