import unittest

from arithmetic_compressor.util import Range
import arithmetic_compressor.arithmetic_coding as AE


class TestArithmeticCoder(unittest.TestCase):
  def setUp(self):
    self.data = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,]
    self.probability = {0: Range(0, 2048), 1: Range(2048, 4096)}

  def test_encode(self):
    encoder = AE.Encoder()
    for bit in self.data:
      encoder.update(self.probability, bit)
    encoder.finish()
    self.assertGreater(len(encoder.get_encoded()), 0,
                       "Encoding data works correctly")

  def test_decode(self):
    encoder = AE.Encoder()
    for bit in self.data:
      encoder.update(self.probability, bit)
    encoder.finish()

    decoder = AE.Decoder(encoder.get_encoded())
    d = []
    for i in range(len(self.data)):
      d += [decoder.decode_symbol(self.probability)]

    self.assertEqual(self.data, d, "Decoding data works correctly")


if __name__ == '__main__':
  unittest.main()
