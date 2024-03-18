# defines activation functions and a few aliases
from .DSA import *
from .DSA.tensorfuncs import MLVecOps as MLVO
from .DSA.tensorfuncs import MLMatrixOps as MLMO
from .DSA.tensorfuncs import MLTensorOps as MLTO

alias E: Float32 = 2.71828174591064453125

########################################################################################################
# defines function for sum used in softmax
@always_inline
fn smsum(vec: MLT, temp: Float32) -> Float32:
  var val: Float32 = 0
  for j in range(vec.dim(0)):
    val += E ** (temp * vec[j])
  return val
########################################################################################################

struct Activation:
  ########################################################################################################
  alias activation_fn = fn(borrowed x: Float32, borrowed a: Float32) -> Float32
  alias vec_activation_fn = fn(borrowed x: MLT, borrowed a: Float32) raises escaping-> MLT
  ########################################################################################################
  alias RELU: Int16 = 0
  alias LEAKYRELU: Int16 = 1
  alias ELU: Int16 = 2
  alias SIGMOID: Int16 = 3
  alias SOFTMAX: Int16 = 4
  alias TANH: Int16 = 5
  ########################################################################################################

  ########################################################################################################
  # defines relu activation function
  @staticmethod
  @always_inline
  fn float_relu(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return x if x > 0 else 0

  alias relu = MLVO.vector_applicable(Self.float_relu)

  @staticmethod
  @always_inline
  fn float_relu_prime(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return 1 if x > 0 else 0

  alias relu_prime = MLVO.vector_applicable(Self.float_relu_prime)
  ########################################################################################################

  ########################################################################################################
  # defines softmax activation function
  @staticmethod
  @always_inline
  fn softmax(borrowed x: MLT, borrowed a: Float32 = 1) raises escaping -> MLT:
    var val = MLT()
    var sum = smsum(x, a)
    for i in range(x.dim(0)):
      MLVO.append(val, 
      E ** (a * x[i]) / sum
      )

    return val

  @staticmethod
  @always_inline
  fn softmax_prime(borrowed x: MLT, borrowed a: Float32 = 1) raises escaping-> MLT:
    var val = MLT()
    return val

  ########################################################################################################

  ########################################################################################################
  # defines tanh activation function
  @staticmethod
  @always_inline
  fn float_tanh(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return (E**x-E**(-x))/(E**x+E**(-x))

  alias tanh = MLVO.vector_applicable(Self.float_tanh)

  @staticmethod
  @always_inline
  fn float_tanh_prime(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return 1-(Self.float_tanh(x)**2)

  alias tanh_prime = MLVO.vector_applicable(Self.float_tanh_prime)

  ########################################################################################################

  ########################################################################################################
  # defines leakyrelu activation function
  @staticmethod
  @always_inline
  fn float_leakyrelu(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return x if x > 0 else a*x

  alias leakyrelu = MLVO.vector_applicable(Self.float_leakyrelu)

  @staticmethod
  @always_inline
  fn float_leakyrelu_prime(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return 1 if x > 0 else a

  alias leakyrelu_prime = MLVO.vector_applicable(Self.float_leakyrelu_prime)

  ########################################################################################################

  ########################################################################################################
  # defines elu activation function
  @staticmethod
  @always_inline
  fn float_elu(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return x if x > 0 else a*(E**x-1)

  alias elu = MLVO.vector_applicable(Self.float_elu)

  @staticmethod
  @always_inline
  fn float_elu_prime(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return 1 if x > 0 else a*(E**x)

  alias elu_prime = MLVO.vector_applicable(Self.float_elu_prime)

  ########################################################################################################

  ########################################################################################################
  # defines sigmoid activation function
  @staticmethod
  @always_inline
  fn float_sigmoid(borrowed x: Float32,borrowed a: Float32 = 1) -> Float32:
    return 1/(1+E**(-x))

  alias sigmoid = MLVO.vector_applicable(Self.float_sigmoid)

  @staticmethod
  @always_inline
  fn float_sigmoid_prime(borrowed x: Float32, a: Float32 = 1) -> Float32:
    return (1-Self.float_sigmoid(x))*Self.float_sigmoid(x)

  alias sigmoid_prime = MLVO.vector_applicable(Self.float_sigmoid_prime)

  ########################################################################################################

  @staticmethod
  @always_inline
  fn evaluate(activation: Int16, x: MLT, a: Float32 = 1) raises -> MLT:
    if activation == Self.RELU:
      return Self.relu(x, a)
    elif activation == Self.LEAKYRELU:
      return Self.leakyrelu(x, a)
    elif activation == Self.ELU:
      return Self.elu(x, a)
    elif activation == Self.SIGMOID:
      return Self.sigmoid(x, a)
    elif activation == Self.SOFTMAX:
      return Self.softmax(x, a)
    elif activation == Self.TANH:
      return Self.tanh(x, a)
    return x

  @staticmethod
  @always_inline
  fn prime_evaluate(activation: Int16, x: MLT, a: Float32 = 1) raises -> MLT:
    if activation == Self.RELU:
      return Self.relu_prime(x, a)
    elif activation == Self.LEAKYRELU:
      return Self.leakyrelu_prime(x, a)
    elif activation == Self.ELU:
      return Self.elu_prime(x, a)
    elif activation == Self.SIGMOID:
      return Self.sigmoid_prime(x, a)
    elif activation == Self.SOFTMAX:
      return Self.softmax_prime(x, a)
    elif activation == Self.TANH:
      return Self.tanh_prime(x, a)
    return x

struct Loss:

  alias loss_func = fn(MLT, MLT) -> Float32

  @staticmethod
  fn mse(y_true: MLT, y_pred: MLT) -> Float32:
    return 3.345# MLTO.mean((y_true - y_pred) ** 2)

  @staticmethod
  fn mse_prime(y_true: MLT, y_pred: MLT)raises -> MLT:
    return 2 * (y_pred - y_true) / y_pred
