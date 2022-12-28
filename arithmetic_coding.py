import collections

# Adapted from https://github.com/nayuki/Reference-arithmetic-coding/blob/master/python/arithmeticcoding.py

STATE_BITS_SIZE = 16


class ArithmeticCoderBase:

  def __init__(self, numbits):
    if numbits < 1:
      raise ValueError("State size out of range")

    self.num_state_bits = numbits
    self.full_range = 1 << self.num_state_bits
    self.half_range = self.full_range >> 1  # Non-zero
    self.quarter_range = self.half_range >> 1  # Can be zero
    self.minimum_range = self.quarter_range + 2  # At least 2
    self.maximum_total = self.minimum_range
    self.state_mask = self.full_range - 1
    self.low, self.high = 0, self.state_mask

  def update(self, probability, symbol):
    low = self.low
    high = self.high
    if low >= high or (low & self.state_mask) != low or (high & self.state_mask) != high:
      raise AssertionError("Low or high out of range")
    range = high - low + 1
    if not (self.minimum_range <= range <= self.full_range):
      raise AssertionError("Range out of range")

    total = max(rng.high for rng in probability.values())
    symlow = probability[symbol].low
    symhigh = probability[symbol].high
    if symlow == symhigh:
      raise ValueError("Symbol has zero frequency")

    newlow = low + symlow * range // total
    newhigh = low + symhigh * range // total - 1
    self.low = newlow
    self.high = newhigh

    while ((self.low ^ self.high) & self.half_range) == 0:
      self.shift()
      self.low = ((self.low << 1) & self.state_mask)
      self.high = ((self.high << 1) & self.state_mask) | 1

    while (self.low & ~self.high & self.quarter_range) != 0:
      self.underflow()
      self.low = (self.low << 1) ^ self.half_range
      self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

  def shift(self):
    raise NotImplementedError()

  def underflow(self):
    raise NotImplementedError()


class Encoder(ArithmeticCoderBase):
  def __init__(self, numbits=STATE_BITS_SIZE):
    super(Encoder, self).__init__(numbits)
    self.encoded_data = []
    self.num_underflow = 0

  def get_encoded(self):
    return self.encoded_data

  def encode_symbol(self, probability, symbol):
    self.update(probability, symbol)

  def finish(self):
    self.encoded_data += [1]

  def shift(self):
    bit = self.low >> (self.num_state_bits - 1)
    self.encoded_data += [bit]
    for _ in range(self.num_underflow):
      self.encoded_data += [bit ^ 1]
    self.num_underflow = 0

  def underflow(self):
    self.num_underflow += 1


class Decoder(ArithmeticCoderBase):
  def __init__(self, encoded_data: list, numbits=STATE_BITS_SIZE):
    super(Decoder, self).__init__(numbits)
    # store encoded data in a deque,
    # so we can access(and pop) the top items efficiently
    self.input = collections.deque(encoded_data)
    self.code = 0
    for _ in range(self.num_state_bits):
      self.code = self.code << 1 | self.read_code_bit()

  def decode_symbol(self, probability):
    freq_total = max(rng.high for rng in probability.values())

    range = self.high - self.low + 1
    offset = self.code - self.low
    value = ((offset + 1) * freq_total - 1) // range

    assert value * range // freq_total <= offset
    assert 0 <= value < freq_total

    symbol = 0
    maxLow = None
    for k, r in probability.items():
      if r.low <= value:
        if not maxLow:
          symbol = k
          maxLow = r
        elif r.low > maxLow.low:
          maxLow = r
          symbol = k

    # encode decoded symbol
    self.update(probability, symbol)

    if not (self.low <= self.code <= self.high):
      raise AssertionError("Code out of range")
    return symbol

  def shift(self):
    self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()

  def underflow(self):
    self.code = (self.code & self.half_range) | ((self.code << 1)
                                                 & (self.state_mask >> 1)) | self.read_code_bit()

  def read_code_bit(self):
    temp = self.input.popleft() if len(self.input) else 0
    return temp
