# defines structs describing neural network models
import .layers as lyrs

trait Model:
  fn eval(inout self):
    ...

  fn fit(inout self):
    ...

struct Sequential():

  var layers: Tuple[lyrs.Layer]

  fn __init__(inout self, layers: Tuple[lyrs.Layer]):
    self.layers = layers

  fn eval(inout self, activation_coefficient: Float32=1):
    pass

  fn fit(inout self, learning_rate: Float32):
    pass
