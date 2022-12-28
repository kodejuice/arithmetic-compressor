#!/usr/bin/env py
import random

from util import *
from compress import AECompressor
from models.base_adaptive_model import BaseFrequencyTable, SimpleAdaptiveModel
from models.ppm import PPMModel, MultiPPM
# from models.context_mixing import


# symbols = [(bin(i)[2:])[-2:] for i in range(0b00, 0b11+1)]
# symbols = ['00', '01', '10', '11']
symbols = ['0', '1']
# symbols = ['a', 'b', 'c']
# symbols = ['00', '01', '10', '11']
prob = {s: 1/len(symbols) for s in symbols}


def init_model():
  model = BaseFrequencyTable(prob)
  # model = SimpleAdaptiveModel(prob)
  # model = PPMModel(prob, 4)
  # model = MultiPPM(prob, 8)
  AEC = AECompressor(model)
  return AEC


def run():
  low_entropy = []
  C = 0
  for N in range(2**16 - 1):
    # AEC = init_model()
    num = bin(N)[2:]
    num = (16-len(num))*"0" + num
    A = h(num)
    if A <= 15:
      # if AEC.compress(num).__len__() < 15:
      low_entropy += [num]
    else:
      C += 1
  print('low entropy count:', len(low_entropy), f"vs [{C}]")

  BC = 80
  print(BC*16)

  for i in range(100000):
    AEC = init_model()

    n = ""
    for __ in range(BC):
      n += random.choice(low_entropy)
    # n = bin(mbit(1440))[2:]

    # n = grams(n, len(symbols[0]))
    L = AEC.compress(n).__len__()
    if L >= len(n):
      print(f"[{i}]. len({len(n)})", h(n), L, "\n", n)
    # print(h(n), "\n", L, " |", "".join(n))
    # print(AEC.model.weights, "\n")


run()

# AEC = init_model()
# D = "000010100000000101100101110111111000100010001001110100000000100111101101011001111110011111011011011000000101000111101100101111010000100001001100111101100011011101011110111100110000011100110000011111100111101010011111110011100010010110000001101000000001110011111001110010110011011111101011000000011001101010110110110101110100010100100010111111111101001011011111011110000111111010011011010000110010000100011000010101001001001111111101010101000000100000100110010010000001100000011010111011001011110101000010010100100010111011101111000100101100000011011101111111011001001001000010101110111101110001000000110001100010010101010000111010101111110001110011000000001111101100001111111011111110011000010110111111110000110000000100110100001000100010000010110000101001111111110011101010010100000001111011111111001101010011111101010011000101000011110101110011110110001110000000000000100101010111111000010111111100111001110111010000100011100000111101100111111110101011111111011011011111011110010001100010001010010001100000111011100110101111010001110111111111111111110011111011111110011001100000100001100010101001100000000011101001000011110101100011110000010100001110010010101000010001000100111000000000100000010101101001100000000110011011101111011010001000001100101110001011111101101111101011011110110011110110101110100111110100111101111100111000011010001000011110110111011101011000001000101110100101110111000100011000010101111111001001110010001101100000"
# # D = "00001010000000010110010111011111100010001000100111010000000010011110110101100111111001111101101101"
# C = AEC.compress(D)
# print(len(C))


# sm = PPMModel(prob)
# ctx = ''
# for c in D:
#   sm.update(c, ctx)
#   ctx += c
# print(sm.cdf())
# print(sm.probability())
# print(sm.entropy(s), h(s))

# print(PPMModel(symbols).scaled_freq())
# print('[Base]')
# BaseFrequencyTable(prob).test_model(False)
# print('[Simple Adaptive]')
# SimpleAdaptiveModel(prob).test_model(False)

# print('[PPM]')
# PPMModel(prob,k=1).test_model(False)
# MultiPPM(prob, models=2).test_model(False)


exit()

# compressed = []
# notcompressed = []

# for N in range(2**16 - 1):
#   bmodel = BaseFrequencyTable({'0': 0.5, '1': 0.5})
#   AEC = AECompressor(bmodel)
#   num = bin(N)[2:]
#   num = (16-len(num))*"0" + num

#   num_compressed = AEC.compress(num)
#   zero_count = num.count('0')
#   D = [f"{(num, 'to: '+str(len(num_compressed)), ('0: '+str(zero_count), '1: '+str(16-zero_count)), 'entropy: ' + str(AEC.model.entropy(num))+' / expected: '+str(AEC.model.entropy(len(num))))}"]
#   if len(num_compressed) < len(num):
#     compressed += D
#   else:
#     notcompressed += D


# print("compressed:\n{}\n\nnot compressed:\n{}".format(
#     '\n'.join(compressed), '\n'.join(notcompressed)))
