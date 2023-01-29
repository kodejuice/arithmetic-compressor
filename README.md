# Arithmetic Coding Library

[![Run tests](https://github.com/kodejuice/arithmetic-compressor/actions/workflows/tests.yml/badge.svg)](https://github.com/kodejuice/arithmetic-compressor/actions/workflows/tests.yml)

This library is an implementation of the [Arithmetic Coding](https://en.wikipedia.org/wiki/Arithmetic_coding) algorithm in Python, along with adaptive statistical data compression models like [PPM (Prediction by Partial Matching)](https://en.wikipedia.org/wiki/Prediction_by_partial_matching), [Context Mixing](en.wikipedia.org/wiki/Context_mixing) and Simple Adaptive models.

## Installation

To install the library, you can use pip:

```bash
pip install arithmetic_compressor
```

## Usage

```python
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel

# create the model
model = StaticModel({'A': 0.5, 'B': 0.25, 'C': 0.25})

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = "AAAAAABBBCCC"
N = len(data)
compressed = coder.compress(data)

# print the compressed data
print(compressed) # => [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
```

And here's an example of how to decode the encoded data:

```python
decoded = coder.decompress(compressed, N)

print(decoded) # -> ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
```

## API Reference

### Arithmetic Compressing

- [**`AECompressor`**](./arithmetic_compressor/compress.py):
  - **`compress(data: list|str, model: Model) -> List[str]`**: Takes in a string or list representing the data, encodes the data using arithmetic coding then returns a string of bits.

  - **`decompress(encoded_data: List[str], length: int) -> List`**: Takes in an encoded string and the length of the original data and decodes the encoded data.

### Models

In addition to the arithmetic coding algorithm, this library also includes adaptive statistical models that can be used to improve compression.

- [**`StaticModel`**](./arithmetic_compressor/models/static_model.py): A class which implements a static model that doesn't adapt to input data or statistics.
- [**`BaseBinaryModel`**](./arithmetic_compressor/models/base_adaptive_model.py): A class which implements a simple adaptive compression algorithm for binary symbols (0 and 1)
- [**`BaseFrequencyTable`**](./arithmetic_compressor/models/base_adaptive_model.py): This implements a basic adaptive frequency table that incrementally adapts to input data.
- [**`SimpleAdaptiveModel`**](./arithmetic_compressor/models/base_adaptive_model.py): A class that implements a simple adaptive compression algorithm.
- [**`PPMModel`**](./arithmetic_compressor/models/ppm.py): A class that implements the PPM compression algorithm.
- [**`MultiPPM`**](./arithmetic_compressor/models/ppm.py): A class which uses weighted averaging to combine several PPM Models of different orders to make predictions.
- [**`BinaryPPM`**](./arithmetic_compressor/models/binary_ppm.py): A class that implements the PPM compression algorithm for binary symbols (0 and 1).
- [**`MultiBinaryPPM`**](./arithmetic_compressor/models/binary_ppm.py): A class which uses weighted averaging to combine several BinaryPPM models of different orders to make predictions.
- [**`ContextMixing_Linear`**](./arithmetic_compressor/models/context_mixing_linear.py): A class which implements the Linear Evidence Mixing variant of the Context Mixing compression algorithm.
- [**`ContextMixing_Logistic`**](./arithmetic_compressor/models/context_mixing_logistic.py): A class which implements the Neural network (Logistic) Mixing variant of the Context Mixing compression algorithm.

All models implement these common methods:

- **`update(symbol)`**: Updates the models statistics given a symbol
- **`probability()`**: Returns the probability of the next symbol
- **`cdf()`**: Returns a cummulative distribution of the next symbol probabilities
- **`test_model()`**: Tests the efficiency of the model to predict symbols

## Models Usage

A closer look at all the models.

### **Simple Models**

- `BaseFrequencyTable(symbol_probabilities: dict)`
- `SimpleAdaptiveModel(symbol_probabilities: dict, adaptation_rate: float)`

The Simple Adaptive models are models that adapts to the probability of a symbol based on the frequency of the symbol in the data.

Here's an example of how to use the Simple Adaptive models included in the library:

```python
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import\
   BaseFrequencyTable,\
   SimpleAdaptiveModel

# create the model
# model = SimpleAdaptiveModel({'A': 0.5, 'B': 0.25, 'C': 0.25})
model = BaseFrequencyTable({'A': 0.5, 'B': 0.25, 'C': 0.25})

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = "AAAAAABBBCCC"
compressed = coder.compress(data)

# print the compressed data
print(compressed) # => [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]
```

The `BaseFrequencyTable` does an incremental adaptation to adapt to the statistics of the input data while the `SimpleAdaptiveModel` is essentially an [exponential moving average](https://en.wikipedia.org/wiki/Moving_average) that adapts to input data relative to the `adaptation_rate`.

### **PPM models**

> <https://en.wikipedia.org/wiki/Prediction_by_partial_matching>

- `PPMModel(symbols: list, context_size: int)`
- `MultiPPMModel(symbols: list, models: int)`
- `BinaryPPMM(context_size: int)`
- `MultiBinaryPPMM(models: int)`

PPM (Prediction by Partial Matching) models are a type of context modeling that uses a set of previous symbols to predict the probability of the next symbol.
Here's an example of how to use the PPM models included in the library:

```python
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import\
   PPMModel,\
   MultiPPM

# create the model
model = PPMModel(['A', 'B', 'C'], k = 3) # no need to pass in probabilities, only symbols

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = "AAAAAABBBCCC"
compressed = coder.compress(data)

# print the compressed data
print(compressed) # => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]
```

The **`MultiPPM`** model uses [weighted averaging](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean) to combine predictions from several PPM models to make a prediction, gives better compression when the input is large.

```python
# create the model
model = MultiPPM(['A', 'B', 'C'], models = 4) # will combine PPM models with context sizes of 0 to 4

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = "AAAAAABBBCCC"
compressed = coder.compress(data)

# print the compressed data
print(compressed) # => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
```

#### **Binary version**

The Binary PPM models **`BinaryPPM`** and **`MultiBinaryPPM`** behave just like normal PPM models demonstrated above, except that they only work for binary symbols `0` and `1`.

```python
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import\
   BinaryPPM,\
   MultiBinaryPPM

# create the model
model = BinaryPPM(k = 3)

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]
compressed = coder.compress(data)

# print the compressed data
print(compressed) # => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]
```

Likewise the **`MultiBinaryPPM`** will combine several Binary PPM models to make prediction using weighted averaging.

### **Context Mixing models**

- `ContextMix_Linear(models: List)`
- `ContextMix_Logistic(learnung_rate: float)`

Context mixing is a type of data compression algorithm in which the next-symbol predictions of two or more statistical models are combined to yield a prediction that is often more accurate than any of the individual predictions.

Two general approaches have been used, linear and logistic mixing. Linear mixing uses a weighted average of the predictions weighted by evidence.
While the logistic (or neural network) mixing first transforms the predictions into the logistic domain, log(p/(1-p)) before averaging.

The library contains a minimal implementation of the algorithm, only the core algorithm is implemented, it doesn't include as many contexts / models as in [PAQ](https://en.wikipedia.org/wiki/PAQ).

> _Note: They only work for binary symbols (**0** and **1**)._

#### **Linear Mixing**

> <https://en.wikipedia.org/wiki/Context_mixing#Linear_Mixing>

The mixer computes a probability by a weighted summation of the N models.

```python
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import ContextMix_Linear

# create the model
model = ContextMix_Linear()

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
compressed = coder.compress(data)

# print the compressed data
print(compressed) # => [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
```

The Linear Mixing model lets you combine other models:

```python
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import ContextMix_Linear,\
   SimpleAdaptiveModel,\
   PPMModel,\
   BaseFrequencyTable

# create the model
model = ContextMix_Linear([
  SimpleAdaptiveModel({0: 0.5, 1: 0.5}),
  BaseFrequencyTable({0: 0.5, 1: 0.5}),
  PPMModel([0, 1], k = 10)
])

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
compressed = coder.compress(data)

# print the compressed data
print(compressed) # => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

#### **Logistic (Neural Network) Mixing**

> <https://en.wikipedia.org/wiki/PAQ#Neural-network_mixing>

A neural network is used to combine models.

```python
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import ContextMix_Logistic

# create the model
model = ContextMix_Logistic()

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
compressed = coder.compress(data)

# print the compressed data
print(compressed) # => [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
```

> _**Note**: This library is intended for learning and educational purposes only. The implementation may not be optimized for performance or memory usage and may not be suitable for use in production environments._
> _Please consider the performance and security issues before using it in production._
> _Please also note that you should thoroughly test the library and its models with real-world data and use cases before deploying it in production._

## More Examples

You can find more detailed examples in the [`/examples`](./examples/) folder in the repository. These examples demonstrate the capabilities of the library and show how to use the different models.

## Contribution

Contributions are very much welcome to the library. If you have an idea for a new feature or have found a bug, please submit an issue or a pull request.

## License

This library is distributed under the MIT License. See the [LICENSE](./LICENSE) file for more information.
