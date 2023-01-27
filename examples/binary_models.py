import sys
sys.path.append('..')

from arithmetic_compressor import binary_ppm
from arithmetic_compressor import AECompressor, util


# 1
def base_binary_model():
  data = [0]*90+[1]*70
  N = len(data)
  entropy = round(util.h(data))
  print('\n==Base Binary Model==')
  print(f"To compress: '{''.join(map(str, data))}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = binary_ppm.BaseBinaryModel(8)
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert (arit_coder.decompress(encoded, N) == data)


# 2
def binary_PPM():
  data = [0]*100 + [1]*500 + [0]*100 + [1]*300
  N = len(data)
  entropy = round(util.h(data))
  print('\n==Binary PPM==')
  print(f"To compress: '{''.join(map(str, data))}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = binary_ppm.BinaryPPM(2)
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert (arit_coder.decompress(encoded, N) == data)


# 3
def multi_binary_PPM():
  data = [0]*100 + [1]*500 + [0]*100 + [1]*300
  N = len(data)
  entropy = round(util.h(data))
  print('\n==Multi Binary PPM==')
  print(f"To compress: '{''.join(map(str, data))}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = binary_ppm.MultiBinaryPPM(10)
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert (arit_coder.decompress(encoded, N) == data)


if __name__ == '__main__':
  base_binary_model()
  binary_PPM()
  multi_binary_PPM()
