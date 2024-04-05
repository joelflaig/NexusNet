# defines structs describing neural network models
from .layers import Layer, Activation
import .layers as lyrs
from .funcs import Activations as Act
from .DSA import Array
from utils.variant import Variant

trait Model:

  fn build(inout self) raises:
    ...

  fn eval(inout self, activation_coefficient: Float32):
    ...

  fn fit(inout self, learning_rate: Float32):
    ...

struct Sequential[size: Int](Model):

  var data: Pointer[Pointer[Layer]]
  var _current: Int

  fn __init__(inout self):
    self.data = Pointer[Pointer[Layer]]()
    self.data = self.data.alloc(size)
    self._current = 0

  @always_inline
  fn __iadd__(inout self, owned lyr: Layer) raises:
    '''
    Adds a `Layer` value at runtime (`Layer` value can be of `Activation` type).
    Function is `@always_inline`.
    '''
    self.data[self._current] = Pointer[Layer].address_of(lyr)
    self._current += 1

  fn add(inout self, owned lyr: Layer) raises:
    '''
    Adds a `Layer` value at runtime (`Layer` value can be of `Activation` type).
    '''
    self.data[self._current] = Pointer[Layer].address_of(lyr)
    self._current += 1
 
  # fn add[activation: Int16 = Act.RELU,
      # activation_coefficient: Float32 = 1
         # ](inout self, owned lyr: Layer = lyrs.Dense) raises:
    # '''
    # Adds a `Layer` and an `Activation` at runtime.
    # '''
    # self += lyr
    ## ain't workin
    # self += Activation[activation, activation_coefficient]()

  fn build(inout self) raises:
    pass

  fn eval(inout self, activation_coefficient: Float32=1):
    pass

  fn fit(inout self, learning_rate: Float32):
    pass

