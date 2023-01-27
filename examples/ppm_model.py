import sys
sys.path.append('..')

from arithmetic_compressor import AECompressor, util
from arithmetic_compressor import ppm

# 1
def ppm_model():
  data = [2,1,0]*150
  N = len(data)
  entropy = round(util.h(data))
  print('\n==PPM==')
  print(f"To compress: '{''.join(map(str, data))}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = ppm.PPMModel([0, 1, 2], True, 1)
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert (arit_coder.decompress(encoded, N) == data)


# 2
def multi_ppm_model():
  data = [1,1,2,0,3,3]*150
  N = len(data)
  entropy = round(util.h(data))
  print('\n==Multi PPM==')
  print(f"To compress: '{''.join(map(str, data))}' (len={len(data)})")
  print(f"Information content(entropy): {entropy}")

  model = ppm.MultiPPM([0, 1, 2,3], 9)
  arit_coder = AECompressor(model)

  encoded = arit_coder.compress(data)
  ratio = (1 - len(encoded)/entropy) * 100
  print(f"Compressed: {encoded} (len={len(encoded)})")
  print(f"Compression ratio: {ratio}%")

  assert (arit_coder.decompress(encoded, N) == data)


if __name__ == '__main__':
  ppm_model()
  multi_ppm_model()
