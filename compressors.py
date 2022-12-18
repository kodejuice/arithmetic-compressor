import copy

import rans as rANS
import arithmeticcoding as AE


# ANS Encoder
class rANSCompressor:
  def __init__(self, model) -> None:
    # clone model, so we dont mutate original
    self.model = copy.deepcopy(model)
    self.__model = copy.deepcopy(model)  # for decoding

  def compress(self, data):
    encoder = rANS.Encoder()
    # encode symbols in reverse order
    # data = data[::-1]

    # context, all symbols encoded before the current symbol
    context = []
    for symbol in data:
      # Use the model to predict the probability of the next symbol
      probability = self.model.probability(context, True)

      # encode the symbol
      encoder.encode_symbol(probability, self.model._symbol_index[symbol])
      # print(symbol, probability)

      # update the model with the new symbol
      # self.model.update(symbol)

      context += [symbol]

    encoded = encoder.get_encoded()
    return list(encoded)

  def decompress(self, encoded_data, length_encoded):
    decoder = rANS.Decoder(encoded_data)
    model = self.__model
    decoded_data = ""
    context = ""
    # print("DECCC")
    for _ in range(length_encoded):
      probability = model.probability(context, True)

      symbol = model.symbols[decoder.decode_symbol(probability)]
      # print(symbol, probability)

      # update model
      # model.update(symbol)

      decoded_data += symbol
      context += symbol
    return decoded_data[::-1]


# Arithmetic Encoder
class AECompressor:
  def __init__(self, model, adapt=True) -> None:
    self.adapt = adapt
    # clone model, so we dont mutate original
    self.model = copy.deepcopy(model)
    self.__model = copy.deepcopy(model)  # for decoding

  def compress(self, data):
    encoder = AE.Encoder()

    # context, all symbols encoded before the current symbol
    context = ""
    for symbol in data:
      # Use the model to predict the probability of the next symbol
      probability = self.model.cdf(context)

      # encode the symbol
      encoder.encode_symbol(probability, symbol)

      if self.adapt:
        # update the model with the new symbol
        self.model.update(symbol, context)

      # add encoded symbol to context
      context += symbol
    encoder.finish()
    return encoder.get_encoded()

  def decompress(self, encoded, length_encoded):
    decoded = ""
    context = ""
    model = self.__model
    decoder = AE.Decoder(encoded)
    for _ in range(length_encoded):
      # probability of the next symbol
      probability = model.cdf(context)

      # decode symbol
      symbol = decoder.decode_symbol(probability)

      if self.adapt:
        # update model
        model.update(symbol, context)

      decoded += symbol
      context += symbol
    return decoded
