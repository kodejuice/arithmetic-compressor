#!/usr/bin/env py
from compressors import AECompressor, rANSCompressor
from misc import *
from adaptive_models import BaseFrequencyTable, PPMModel, MultiPPM

sc = 10
prob = {str(s): 1/sc for s in range(0, sc)}

# bmodel = BaseFrequencyTable(prob)
pmodel = PPMModel(prob, 4)
mpmodel = MultiPPM(prob)

# data = bin(mbit(100000))[2:]
# data = generate_data(prob, 100000, shuffle=True)
data = "000012344445677789"*5000
data += "009937754421110888"*5000
# data = shuffle_string(data)

# AE
AEC = AECompressor(pmodel)
print('AE', len(data))
ans_encoded = AEC.compress(data)
ans_decoded = AEC.decompress(ans_encoded, len(data))
print(len(ans_encoded),
      f"[entropy: {AEC.model.entropy(data)} | expected: {AEC.model.entropy(len(data))}]")
print(data == ans_decoded)
print(AEC.model.probability())

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

compressed = []
notcompressed = []

for N in range(2**16 - 1):
  bmodel = BaseFrequencyTable({'0': 0.5, '1': 0.5})
  AEC = AECompressor(bmodel)
  num = bin(N)[2:]
  num = (16-len(num))*"0" + num

  num_compressed = AEC.compress(num)
  zero_count = num.count('0')
  D = [f"{(num, 'to: '+str(len(num_compressed)), ('0: '+str(zero_count), '1: '+str(16-zero_count)), 'entropy: ' + str(AEC.model.entropy(num))+' / expected: '+str(AEC.model.entropy(len(num))))}"]
  if len(num_compressed) < len(num):
    compressed += D
  else:
    notcompressed += D


print("compressed:\n{}\n\nnot compressed:\n{}".format(
    '\n'.join(compressed), '\n'.join(notcompressed)))
