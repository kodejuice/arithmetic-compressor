import sys
sys.path.append('..')

from arithmetic_compressor import AECompressor, util
from arithmetic_compressor import static_model

# 1
def sample1():
  data = "a"*180 + "b"*20
  N = len(data)
  entropy = round(util.h(data))
  print(f"\nTo compress: '{data}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = static_model.StaticModel({'a': 0.9, 'b': 0.1})
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert ("".join(arit_coder.decompress(encoded, N)) == data)


# 2
def sample2():
  data = "a"*70+"b"*25+"c"*5
  N = len(data)
  entropy = round(util.h(data))
  print(f"\nTo compress: '{data}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = static_model.StaticModel({'a': 0.7, 'b': 0.25, 'c': 0.05})
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert ("".join(arit_coder.decompress(encoded, N)) == data)


if __name__ == '__main__':
  print('Compressing with static model')
  sample1()
  sample2()
