import copy
import arithmetic_coding as AE

# Arithmetic Encoder
class AECompressor:
  def __init__(self, model, adapt=True) -> None:
    self.adapt = adapt
    # clone model, so we dont mutate original
    self.model = model
    # self.model = copy.deepcopy(model)
    # self.__model = copy.deepcopy(model)  # for decoding

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
      if len(context) == 10:
        context = context[1:]
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
