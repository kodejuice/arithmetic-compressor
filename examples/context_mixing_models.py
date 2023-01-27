import sys
sys.path.append('..')

from arithmetic_compressor import AECompressor, util
from arithmetic_compressor import context_mixing_linear, context_mixing_logistic

# 1
def context_mixing_linear_model():
  data = [0]*150+[1]*500
  N = len(data)
  entropy = round(util.h(data))
  print('\n==Context Mixing (Linear)==')
  print(f"To compress: '{data}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = context_mixing_linear.ContextMix_Linear()
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert arit_coder.decompress(encoded, N) == data


# 2
def context_mixing_logistic_model():
  data = [0]*150+[1]*500
  N = len(data)
  entropy = round(util.h(data))
  print('\n==Context Mixing (Neural Network / Logistic)==')
  print(f"To compress: '{data}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = context_mixing_logistic.ContextMix_Logistic(0.1)
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert arit_coder.decompress(encoded, N) == data


if __name__ == '__main__':
  context_mixing_linear_model()
  context_mixing_logistic_model()
