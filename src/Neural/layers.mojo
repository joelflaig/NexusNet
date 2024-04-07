# defines layer trait and structs conforming to it
from .funcs import Activations as Act
from .DSA import MLT, f32
from .DSA.tensorfuncs import MLTensorOps as MLTO
from random import randn
from tensor import Tensor, TensorShape

# Layer trait
trait Layer(CollectionElement):
  # CollectionElement necessary to allow use in Variant

  '''
  A trait made for the programmers convenience.
  '''
  fn __copyinit__(inout self, borrowed other: Self) -> None:
    ...


struct Flatten(Layer, CollectionElement):
  # second CollectionElement necessary because otherwise not recognized as CollectionElement

  var in_shape: MLT

  fn __init__(inout self, in_shape: MLT):
    self.in_shape = in_shape

  fn __copyinit__(inout self, borrowed other: Self):
    self.in_shape = other.in_shape

  fn __moveinit__(inout self, owned other: Self):
    self.in_shape = other.in_shape

  @always_inline
  fn eval(inout self, inout input: MLT) raises -> MLT:
    return MLTO.flatten(input)


struct Activation[
    func: Int16 = Act.RELU, 
    func_coefficient: Float32 = 1
    ](Layer, CollectionElement): 
  # second CollectionElement necessary because otherwise not recognized as CollectionElement
 
  var input: MLT
 
  fn __init__(inout self):
    self.input = MLT()

  fn __copyinit__(inout self, borrowed other: Self):
    self.input = other.input

  fn __moveinit__(inout self, owned other: Self):
    self.input = other.input

  @always_inline
  fn eval(self, input: MLT) raises -> MLT:
    return Act.evaluate[func](input, func_coefficient)

  # make `a` a field
  @always_inline
  fn fit(self, owned out_gradient: MLT, owned learning_rate: Float32) raises -> MLT:
    var pe = Act.prime_evaluate[func](self.input, func_coefficient)
    return MLTO.hadamard(
        out_gradient, 
        pe
        )


struct Dense(Layer, CollectionElement):
  # second CollectionElement necessary because otherwise not recognized as CollectionElement

  var input: MLT
  var neurons: Int 
  var weights: MLT
  var biases: MLT

  fn __init__(inout self,
      neurons: Int,
      input_size: Int,
      rand_init_mean: Float64=0.5,
      rand_init_variance: Float64=0.5,
      zeroes: Bool=False) raises:

    self.input = MLT(input_size)
    self.neurons = neurons
    self.weights = MLT(neurons, input_size)
    self.biases = MLT(neurons)

    if zeroes:
      pass
    else:
      randn[f32](self.weights.data(), neurons * input_size, rand_init_mean, rand_init_variance)
      randn[f32](self.biases.data(), neurons, rand_init_mean, rand_init_variance)

  fn __copyinit__(inout self, borrowed other: Self):
    self.input = other.input
    self.neurons = other.neurons
    self.weights = other.weights
    self.biases = other.biases

  fn __moveinit__(inout self, owned other: Self):
    self.input = other.input
    self.neurons = other.neurons
    self.weights = other.weights
    self.biases = other.biases

  fn eval(inout self, input: MLT) raises -> MLT:
    if input.spec() != self.input.spec():
      raise Error("Size of input does not match input size at initialization.")

    self.input = input
    return self.biases + MLTO.dot(self.weights, self.input)

  fn fit(inout self, out_gradient: MLT, learning_rate: Float32) raises -> MLT:
    self.weights = self.weights - learning_rate * MLTO.dot(out_gradient, self.input)
    self.biases = self.biases - learning_rate * out_gradient
    return MLTO.dot(self.weights, out_gradient)


struct Conv2D(Layer, CollectionElement):
  # second CollectionElement necessary because otherwise not recognized as CollectionElement

  var features: Int 
  var kernels: MLT
  var biases: MLT

  fn __init__(inout self, features: Int) raises:
    
    self.features = features
    self.kernels = MLT(); # self.kernels.random_tensor3d(features, 0)
    self.biases = MLT(); # self.biases.random_tensor3D(features)

  fn __copyinit__(inout self, borrowed other: Self):
    self.features = other.features
    self.kernels = other.kernels
    self.biases = other.biases

  fn __moveinit__(inout self, owned other: Self):
    self.features = other.features
    self.kernels = other.kernels
    self.biases = other.biases

  fn eval(inout self, input: MLT) raises:
    pass
