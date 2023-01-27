import sys
sys.path.append('..')

from arithmetic_compressor import AECompressor, util
from arithmetic_compressor import base_adaptive_model

# 1
def basic_frequency_table():
  data = "a"*150+"b"*50+"c"*25+"d"*12
  N = len(data)
  entropy = round(util.h(data))
  print('\n==Basic Frequency Table==')
  print(f"To compress: '{data}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = base_adaptive_model.BaseFrequencyTable({'a': .25, 'b': .25, 'c': .25, 'd': .25})
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert ("".join(arit_coder.decompress(encoded, N)) == data)


# 2
def simple_adaptive_model():
  data = "aabbabbaababbccbbdbcbdcdbcd"*3
  N = len(data)
  entropy = round(util.h(data))
  print('\n==Simple Adaptive Model==')
  print(f"To compress: '{data}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = base_adaptive_model.SimpleAdaptiveModel({'a': .25, 'b': .25, 'c': .25, 'd': .25})
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert ("".join(arit_coder.decompress(encoded, N)) == data)


if __name__ == '__main__':
  print("Compressing with simple adaptive models")
  basic_frequency_table()
  simple_adaptive_model()
