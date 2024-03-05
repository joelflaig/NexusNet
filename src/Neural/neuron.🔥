# defines the Neuron trait and structs conforming to the Neuron trait
import .funcs as act
from .datastruct import MLVec, f32
from random import rand

'''
This module is intended for use
in evolutionary simulations,
for example biosim4 
(https://github.com/davidrmiller/biosim4)
'''

# Neuron trait
trait Neuron(CollectionElement):

  '''
  This trait is planned as a interface 
  to programm own Neuron structs and classes.
  '''

  fn __init__(inout self) raises:
    ...

  fn __copyinit__(inout self, borrowed other: Self):
    ...
    
  fn __moveinit__(inout self, owned other: Self):
    ...

  fn eval(inout self, a: Float32 = 1) raises:
    ...

# Normcell struct conforming to Neuron trait
struct Cell(Neuron):

  '''
  This struct is intended to be inhereted 
  by own custom structs.
  '''

  var input: MLVec
  var weights: MLVec
  var bias: Float32
  var value: Float32
  var activation: act.activation_fn

  fn  __init__(inout self) raises:
    self.value = 0
    self.input = MLVec()
    self.activation = act.float_relu
    self.weights = MLVec()
    self.bias = 0


  fn __init__(inout self, input: MLVec, activation: act.activation_fn) raises:
    self.value = 0
    self.input = input
    self.activation = activation
    self.weights = MLVec(); self.weights.random_vector(len(input))
    self.bias = rand[f32](1)[0]
  
  fn __moveinit__(inout self, owned other: Self):
    self.input = other.input
    self.weights = other.weights
    self.bias = other.bias
    self.value = other.value
    self.activation = other.activation

  fn __copyinit__(inout self, borrowed other: Self):
    self.input = other.input
    self.weights = other.weights
    self.bias = other.bias
    self.value = other.value
    self.activation = other.activation

  fn eval(inout self, a: Float32 = 1) raises:
    self.value = self.activation(self.input * self.weights + self.bias, a)
