import math
import random
from random import randint as R


def rand_arr(n, a, b):
  return [R(a, b) for i in range(n)]


# return randon m bit number
def mbit(m):
  return R(2**(m - 1), 2**m - 1)


# partition string into grams of length n
def grams(s, n=1):
  r = []
  while s:
    r += [s[:n]]
    s = s[n:]
  return r


# entropy
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

  # expected number of bits per symbol
  for symbol in X:
    h += X[symbol] * -math.log2(X[symbol])

  # if N > 1
  #  then this is the expected number of bits for N symbols
  return h * N


def HF(freq_table: dict):
  """compute entropy from a frequency table

  Args:
      freq_table (dict): frequency table
  """
  h, N = 0, sum(freq_table.values())
  for sym, freq in freq_table.items():
    h += freq * -math.log2(freq/N)
    # h += sum([-math.log2(freq/N)] * freq)
  return h


# entropy from given string
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


# def feistel(data, key, rounds):
#     # Split the data into left and right halves
#     left = data[:len(data) // 2]
#     right = data[len(data) // 2:]

#     # Perform the specified number of rounds
#     for i in range(rounds):
#         # XOR the right half with the round key
#         right = right ^ key

#         # Swap the left and right halves
#         left, right = right, left

#     # Concatenate the left and right halves to form the encrypted data
#     return left + right

# def decrypt(data, key, rounds):
#     # Split the data into left and right halves
#     left = data[:len(data) // 2]
#     right = data[len(data) // 2:]

#     # Perform the specified number of rounds
#     for i in range(rounds):
#         # Swap the left and right halves
#         left, right = right, left

#         # XOR the right half with the round key
#         right = right ^ key

#     # Concatenate the left and right halves to form the decrypted data
#     return left + right

# # Encrypt some data
# data = b"Hello, world!"
# key = b"secret"
# encrypted_data = feistel(data, key, 10)

# # Decrypt the data
# decrypted_data = decrypt(encrypted_data, key, 10)

# # Verify that the original data is recovered
# assert data == decrypted_data
