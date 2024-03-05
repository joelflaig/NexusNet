# defines layer trait and structs conforming to it
from .funcs import Activation as act
from .datastruct import MLVec, MLMatrix, MLT3D, TensorWrapper, MDTensorWrapper

# Layer trait
trait Layer():
  '''
  A trait made for the programmers convenience.
  '''
  pass


struct Flatten2D(Layer):
  '''Currently a unified `Flatten` layer is not possible, due to the fact that mojo does not support parametrized traits.'''

  fn eval(inout self, inout input: MLMatrix, a: Float32 = 0) raises -> MLVec:
    return input.flatten()


struct Flatten3D(Layer):
  '''Currently a unified `Flatten` layer is not possible, due to the fact that mojo does not support parametrized traits.'''

  fn eval(inout self, inout input: MLT3D, a: Float32 = 0) raises -> MLVec:
    return input.flatten()


struct Activation[func: Int16](Layer):
  
  var input: MLVec
  
  fn __init__(inout self) raises:
    self.input = MLVec()

  fn eval(inout self, input: MLVec, a: Float32 = 0) raises -> MLVec:
    return act.evaluate(func, input, a)


struct Dense(Layer):

  var input: MLVec
  var neurons: Int 
  var weights: MLMatrix
  var biases: MLVec

  fn __init__(inout self, neurons: Int) raises:

    self.input = MLVec()    
    self.neurons = neurons
    self.weights = MLMatrix(); self.weights.random_matrix(neurons, 0)
    self.biases = MLVec(); self.biases.random_vector(neurons)

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

  fn eval(inout self, input: MLVec, a: Float32 = 0) raises -> MLVec:
    self.input = input
    return self.biases + (input * self.weights)

  fn fit(inout self, out_gradient: MLVec, learning_rate: Float32) raises -> MLVec:
    return MLVec()


struct Conv2D(Layer):

  var features: Int 
  var kernels: MLT3D
  var biases: MLT3D

  fn __init__(inout self, features: Int, owned prec_layer: Layer, act: Int16) raises:
    
    self.features = features
    self.kernels = MLT3D(); # self.kernels.random_tensor3d(features, 0)
    self.biases = MLT3D(); # self.biases.random_tensor3D(features)

  fn __copyinit__(inout self, borrowed other: Self):
    self.features = other.features
    self.kernels = other.kernels
    self.biases = other.biases

  fn __moveinit__(inout self, owned other: Self):
    self.features = other.features
    self.kernels = other.kernels
    self.biases = other.biases

  fn eval(inout self, input: TensorWrapper, a: Float32 = 0) raises:
    pass
