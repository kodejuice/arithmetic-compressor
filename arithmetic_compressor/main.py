#!/usr/bin/env py
import random

from util import *
from compress import AECompressor
from models.base_adaptive_model import BaseFrequencyTable, SimpleAdaptiveModel
from models.ppm import PPMModel, MultiPPM
from models.binary_ppm import MultiBinaryPPM, BinaryPPM, BaseBinaryModel
from models.context_mixing_linear import CharModel, ContextMix_Linear
from models.context_mixing_logistic import ContextMix_Logistic
from models.static_model import StaticModel

# symbols = [(bin(i)[2:])[-2:] for i in range(0b00, 0b11+1)]
# symbols = ['00', '01', '10', '11']
symbols = ['0', '1']
# symbols = ['a', 'b', 'c']
# symbols = ['00', '01', '10', '11']
prob = {s: 1/len(symbols) for s in symbols}


def rand(symbols, n):
  res = ""
  for i in range(n):
    res += random.choice(symbols)
  return res


def init_model(p=None):
  if not p:
    p = prob
  # model = BaseFrequencyTable(p)
  # model = SimpleAdaptiveModel(p)
  # model = PPMModel(p, 4)
  # model = MultiPPM(p, 8)
  # model = MultiBinaryPPM(8)
  # model = BinaryPPM(5)
  # model = ContextMix_Linear()
  model = ContextMix_Linear([
      CharModel(16),
      # BaseBinaryModel(),
      SimpleAdaptiveModel(p),
  ])
  AEC = AECompressor(model)
  return AEC


def run():
  # L = 16
  # low_entropy = []
  # C = 0
  # for N in range(2**L - 1):
  #   # AEC = init_model()
  #   num = bin(N)[2:]
  #   num = (L-len(num))*"0" + num
  #   A = h(num)
  #   if A <= (L-1):
  #     # if AEC.compress(num).__len__() < 15:
  #     low_entropy += [num]
  #   else:
  #     C += 1
  # print('low entropy count:', len(low_entropy), f"vs [{C}]")

  BLOCK_COUNT = 90
  BLK_SIZE = 16
  BITS = BLOCK_COUNT * BLK_SIZE
  print(BITS, "bits")

  n = bin(mbit(BITS))[2:]
  total_entropy_high = 0
  total_entropy_low = 0

  AEC = init_model()
  # B = (list(int(b in low_entropy) for b in g))
  # for b in g:
  #   if b not in low_entropy:
  #     total_entropy_high += h(b)
  #   else:
  #     total_entropy_low += h(b)
  # print(B)
  # print(sum(B))
  # print('total entropy high', total_entropy_high)
  # print('total entropy low', total_entropy_low)
  # print('total', total_entropy_high + total_entropy_low)

  sst = [f"1{'0'*(BLK_SIZE+1)}" for _ in range(len(g))]
  sst_s = "".join(sst)

  print(h(sst_s))
  print(h(n))

  # print(AEC.compress(n).__len__() < BITS)

  # return
  # for i in range(50_000):
  #   AEC = init_model()

  #   n = ""
  #   for __ in range(BLOCK_COUNT):
  #     n += random.choice(low_entropy)
  #   # n = bin(mbit(1440))[2:]

  #   # n = grams(n, len(symbols[0]))
  #   L = AEC.compress(n).__len__()
  #   if L >= len(n):
  #     print(f"[{i}]. len({len(n)})", h(n), L, "\n", n)


# run()

# N = bin(mbit(256))[2:]


cnt = 0
runs = 10
B = 100
A = "012"
P = {s: 1/len(A) for s in A}
# for i in range(runs):
#   AEC = init_model(P)
#   r = rand(A, B)
#   cnt += AEC.compress(r).__len__() < B-1

# print(f"probability: {cnt/runs}")

P = {0: 0.5, 1: 0.5}
aec = init_model(P)
data = [random.choice([0, 1]) for _ in range(200)]
# print(aec.compress(data))


# TEST

prob = {0: 0.9, 1: 0.1}
custom_data = generate_data(prob, 10_000, True)
# model = StaticModel(prob)
# model = BaseFrequencyTable(prob)
# model = ContextMix_Logistic(0.0000001)
# model = ContextMix_Linear()
# aec = AECompressor(model)
# C = aec.compress(custom_data)
# print(aec.decompress(C, len(custom_data)) == custom_data)
# print(len(C))

# StaticModel(prob).test_model(False, custom_data=custom_data)
ContextMix_Linear().test_model(False, custom_data=custom_data)
ContextMix_Logistic(0.1).test_model(False, custom_data=custom_data)
# MultiBinaryPPM().test_model(False, custom_data=custom_data)
# BaseFrequencyTable(prob).test_model(False, custom_data=custom_data)
# ContextMix_Linear([
#   # CharModel(16),
#   BaseBinaryModel(),
#   SimpleAdaptiveModel(prob),
# ]).test_model(False, custom_data=custom_data)
# MultiPPM(prob, models=8).test_model(False, custom_data=custom_data)
# SimpleAdaptiveModel(prob).test_model(False, custom_data=custom_data)


# ContextMix_Linear().test_model(False)
# MultiBinaryPPM().test_model(False)
# MultiPPM(prob, models=6).test_model(False)
# SimpleAdaptiveModel(prob).test_model(False)
