'''
At this time it is not possible to 
define Models in mojo since it does not have a
working Array type.
'''
# defines structs describing neural network models
from .layers import Layer, Activation
from .funcs import Activation as Act
from .DSA import Array

trait Model:
  '''
  At this time it is not possible to 
  define Models in mojo since it does not have a
  working Array type.
  '''
  fn eval(inout self):
    ...

  fn fit(inout self):
    ...

alias ModelStore = Array[Pointer[Layer]]

struct Sequential():
  '''
  This struct is not functional. 
  Use it at your own risk.
  '''

  var lyrs: ModelStore
  var lyrnum: Int
  var _current: Int
  var params: Int

  fn __init__(inout self):
    self.lyrs = ModelStore()
    self.lyrnum = 0
    self._current = 0
    self.params = 0

  fn add(inout self, owned lyr: Layer, owned activation: Int16) raises:
    self.lyrs.append(Pointer[Layer].address_of(lyr))
    var act = Activation(activation)
    # self.lyrs.append(Pointer[Layer].address_of(act))
    self.lyrnum += 1

  fn eval(inout self, activation_coefficient: Float32=1):
    pass

  fn fit(inout self, learning_rate: Float32):
    pass
