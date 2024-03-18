from Neural.layers import Dense
from Neural.funcs import Activation as Act
from Neural.model import Sequential

fn main() raises:
  var model = Sequential()
  model.add(Dense(2, 2), Act.TANH)
  model.add(Dense(3, 2), Act.TANH)
  model.add(Dense(1, 3), Act.TANH)
