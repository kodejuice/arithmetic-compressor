import sys
sys.path.append('..')

import random
from arithmetic_compressor.models import ContextMix_Logistic
from arithmetic_compressor.models import ContextMix_Linear
from arithmetic_compressor.models import StaticModel
from arithmetic_compressor.models import PPMModel, MultiPPM
from arithmetic_compressor.models import BinaryPPM, MultiBinaryPPM, BaseBinaryModel
from arithmetic_compressor.models import BaseFrequencyTable, SimpleAdaptiveModel

# Note: These models can be tuned to give better results
# E.g in the PPM models you could simply increase the context size and you'll get better result
# For the simple adaptive models, you could fine tune the adaptation rate
# For the Logistic model you could also fine tune the learning rate and you'll get great results

def custom_test_result(sequence, probabilities):
  N = len(sequence)
  tests = [
      StaticModel(probabilities).test_model(False, N, sequence),
      BaseFrequencyTable(probabilities).test_model(False, N, sequence),
      SimpleAdaptiveModel(probabilities).test_model(False, N, sequence),
      BaseBinaryModel().test_model(False, N, sequence),
      PPMModel([0, 1]).test_model(False, N, sequence),
      MultiPPM([0, 1]).test_model(False, N, sequence),
      BinaryPPM().test_model(False, N, sequence),
      MultiBinaryPPM().test_model(False, N, sequence),
      ContextMix_Linear().test_model(False, N, sequence),
      ContextMix_Logistic().test_model(False, N, sequence),
  ]
  tests.sort(key=lambda item: item[1])
  return tests


def run_test_random():
  print("\n====Random sequence test====")

  probabilities = {0: 0.5, 1: 0.5}

  tests = [
      StaticModel(probabilities).test_model(),
      BaseFrequencyTable(probabilities).test_model(),
      SimpleAdaptiveModel(probabilities).test_model(),
      BaseBinaryModel().test_model(),
      PPMModel([0, 1]).test_model(),
      MultiPPM([0, 1]).test_model(),
      BinaryPPM().test_model(),
      MultiBinaryPPM().test_model(),
      ContextMix_Linear().test_model(),
      ContextMix_Logistic().test_model(),
  ]

  # sort in increasing order of % error
  tests.sort(key=lambda item: item[1])

  for name, error, output in tests:
    print(output)


def run_test_predictable():
  print("\n====Predictable sequence test====")

  probabilities = {0: 0.5, 1: 0.5}
  sequence = [0]*5_000 + [1]*5_000

  tests = custom_test_result(sequence, probabilities)
  for _, _, output in tests:
    print(output)


def run_test_repeated():
  print("\n====Repeated pattern sequence test====")

  probabilities = {0: 0.5, 1: 0.5}
  sequence = [0, 0, 0, 1, 1] * 2_000

  tests = custom_test_result(sequence, probabilities)
  for _, _, output in tests:
    print(output)


if __name__ == '__main__':
  run_test_random()
  run_test_predictable()
  run_test_repeated()
