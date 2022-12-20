#!/usr/bin/env py
from compressors import AECompressor, rANSCompressor
from misc import *
from adaptive_models import BaseFrequencyTable,\
    BaseBinaryModel,\
    PPMModel,\
    MultiPPM,\
    BinaryPPM,\
    MultiBinaryPPM\

symbols = ['0', '1']
# symbols = ['00', '01', '10', '11']
prob = {s: 1/len(symbols) for s in symbols}


def init_model():
  # model = BaseFrequencyTable(prob)
  # model = PPMModel(prob, 1)
  # model = MultiPPM(prob)
  # model = BaseBinaryModel()
  # model = BinaryPPM()
  model = MultiBinaryPPM()
  AEC = AECompressor(model)
  return AEC


low_entropy = []
C = 0
for N in range(2**16 - 1):
  # AEC = init_model()
  num = bin(N)[2:]
  num = (16-len(num))*"0" + num
  E, A = h(num)
  if A <= 15:
    # if AEC.compress(num).__len__() < 15:
    low_entropy += [num]
  else:
    C += 1
print('low entropy count:', len(low_entropy), f"[{C}]")


BC = 90
print(BC*16)

for _ in range(1000):
  AEC = init_model()

  n = ""
  for _ in range(BC):
    n += random.choice(low_entropy)
  # n = bin(mbit(256))[2:]

  # n = grams(n, len(symbols[0]))
  L = AEC.compress(n).__len__()
  if L > len(n):
    print(h(n), L, "\n", n)
  # print(h(n), "\n", L, " |", "".join(n))
  # print(AEC.model.weights, "\n")


# BinaryPPM().test_model(False)
# BaseBinaryModel().test_model(False)
# PPMModel(['0','1']).test_model(False)
# MultiPPM(['0', '1']).test_model(False)
# MultiBinaryPPM().test_model(False)


# m=MultiBinaryPPM()
# s,ctx='11101010101001100101011001011111001101110100101110101',''
# for c in s:
#   m.update(c, ctx)
#   ctx+=c
# print(m.weights)


# data = bin(mbit(100000))[2:]
# data = generate_data(prob, 100000, shuffle=True)
# data = "000012344445677789"*5000
# data += "009937754421110888"*5000
# data = shuffle_string(data)

# AE
# AEC = AECompressor(mpmodel)
# print('AE', len(data))
# ans_encoded = AEC.compress(data)
# ans_decoded = AEC.decompress(ans_encoded, len(data))
# print(len(ans_encoded),
#       f"[entropy: {AEC.model.entropy(data)} | expected: {AEC.model.entropy(len(data))}]")
# print(data == ans_decoded)
# print(AEC.model.probability())

# print('\n', AEC.model.weights)
# for i in range(len(AEC.model.weights)):
#   print(i, AEC.model.models[i].probability(data[-6:]))


# ANS
# rANSC = rANSCompressor(bmodel)
# print('rANS')
# ans_encoded = rANSC.compress(data)
# print(len(ans_encoded)*32, rANSC.bmodel.entropy(len(data)))
# ans_decoded = rANSC.decompress(ans_encoded, len(data))
# print(ans_decoded, data)
# print(ans_decoded == data)


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
