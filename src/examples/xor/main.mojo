from Neural.layers import Dense, Activation
from Neural.funcs import Activations as Act
from Neural.model import Sequential

fn main() raises:
  var model = Sequential[3]()
  model += Dense(2, 2)
  model += Activation[Act.TANH]()
  model.add(Dense(2, 2), Act.TANH)
  model.add(Dense(3, 2), Act.TANH)
  model.add(Dense(1, 3), Act.TANH)
