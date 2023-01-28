import unittest

import arithmetic_compressor.compress as ae_compress
from arithmetic_compressor.models import static_model, base_adaptive_model, binary_ppm, context_mixing_linear, context_mixing_logistic, ppm


class TestCompress(unittest.TestCase):
  def setUp(self):
    self.data = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
                 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0,]

  def compress_with_model(self, model, data):
    '''Compress with given model and return decompressed result'''
    ae_coder = ae_compress.AECompressor(model)
    return ae_coder.decompress(ae_coder.compress(self.data), len(data))

  def test_static_model(self):
    model = static_model.StaticModel({0: 0.57, 1: 0.43})
    self.assertEqual(self.compress_with_model(model, self.data), self.data)

  def test_base_adaptive_model(self):
    model = base_adaptive_model.SimpleAdaptiveModel({0: 0.5, 1: 0.5})
    self.assertEqual(self.compress_with_model(model, self.data), self.data)

  def test_binary_ppm(self):
    model = binary_ppm.BinaryPPM()
    self.assertEqual(self.compress_with_model(model, self.data), self.data)

  def test_multi_binary_ppm(self):
    model = binary_ppm.MultiBinaryPPM()
    self.assertEqual(self.compress_with_model(model, self.data), self.data)

  def test_base_binary_model(self):
    model = binary_ppm.BaseBinaryModel()
    self.assertEqual(self.compress_with_model(model, self.data), self.data)

  def test_ppm(self):
    model = ppm.PPMModel([0, 1])
    self.assertEqual(self.compress_with_model(model, self.data), self.data)

  def test_multi_ppm(self):
    model = ppm.MultiPPM([0, 1])
    self.assertEqual(self.compress_with_model(model, self.data), self.data)

  def test_context_mixing_linear(self):
    model = context_mixing_linear.ContextMix_Linear()
    self.assertEqual(self.compress_with_model(model, self.data), self.data)

  def test_context_mixing_logistic(self):
    model = context_mixing_logistic.ContextMix_Logistic()
    self.assertEqual(self.compress_with_model(model, self.data), self.data)
