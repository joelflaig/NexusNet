# defines layer trait and structs conforming to it
from .funcs import Activation as act
from .DSA import *
from .DSA.tensorfuncs import MLTensorOps as MLTO
from random import randn
from tensor import TensorShape

# Layer trait
trait Layer():
  '''
  A trait made for the programmers convenience.
  '''
  pass


struct Flatten(Layer):
  @always_inline
  fn eval(inout self, inout input: MLT, a: Float32 = 0) raises -> MLT:
    return MLTO.flatten(input)


struct Activation[func: Int16](Layer):
  
  var input: MLT
  
  fn __init__(inout self) raises:
    self.input = MLT()

  @always_inline
  fn eval(inout self, input: MLT, a: Float32 = 0) raises -> MLT:
    return act.evaluate(func, input, a)


struct Dense(Layer):

  var input: MLT
  var neurons: Int 
  var weights: MLT
  var biases: MLT

  fn __init__(inout self, neurons: Int, input_size: Int, rand_init_mean: Float64=0.5, rand_init_variance: Float64=0.5) raises:

    self.input = MLT(input_size)
    self.neurons = neurons
    self.weights = MLT()
    self.weights = randn[f32](TensorShape(neurons, input_size), rand_init_mean, rand_init_variance)
    self.biases = MLT()
    self.biases = randn[f32](TensorShape(neurons), rand_init_mean, rand_init_variance)

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

  fn eval(inout self, input: MLT, a: Float32 = 0) raises -> MLT:
    if input.spec() != self.input.spec():
      raise Error("Size of input does not match input size at initialization.")

    self.input = input
    return self.biases + (input * self.weights)

  fn fit(inout self, out_gradient: MLT, learning_rate: Float32) raises -> MLT:
    return MLT()


struct Conv2D(Layer):

  var features: Int 
  var kernels: MLT
  var biases: MLT

  fn __init__(inout self, features: Int, owned prec_layer: Layer, act: Int16) raises:
    
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

  fn eval(inout self, input: MLT, a: Float32 = 0) raises:
    pass
