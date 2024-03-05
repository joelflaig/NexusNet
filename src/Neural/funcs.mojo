# defines activation functions and a few aliases
from .datastruct import f32, MLVec

alias E: Float32 = 2.71828174591064453125

########################################################################################################
# defines function for sum used in softmax
@always_inline
fn smsum(vec: MLVec, temp: Float32) -> Float32:
  var val: Float32 = 0
  for j in range(len(vec)):
    val += E ** (temp * vec[j])
  return val
########################################################################################################

struct Activation:
  ########################################################################################################
  alias activation_fn = fn(borrowed x: Float32, borrowed a: Float32) -> Float32
  alias vec_activation_fn = fn(borrowed x: MLVec, borrowed a: Float32) raises escaping-> MLVec
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

  alias relu = MLVec.vector_applicable[f32](Self.float_relu)

  @staticmethod
  @always_inline
  fn float_relu_prime(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return 1 if x > 0 else 0

  alias relu_prime = MLVec.vector_applicable[f32](Self.float_relu_prime)
  ########################################################################################################

  ########################################################################################################
  # defines softmax activation function
  @staticmethod
  @always_inline
  fn softmax(borrowed x: MLVec, borrowed a: Float32 = 1) raises escaping -> MLVec:
    var val = MLVec()
    let sum = smsum(x, a)
    for i in range(len(x)):
      val.append(
      E ** (a * x[i]) / sum
      )

    return val

  @staticmethod
  @always_inline
  fn softmax_prime(borrowed x: MLVec, borrowed a: Float32 = 1) raises escaping-> MLVec:
    pass

  ########################################################################################################

  ########################################################################################################
  # defines tanh activation function
  @staticmethod
  @always_inline
  fn float_tanh(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return (E**x-E**(-x))/(E**x+E**(-x))

  alias tanh = MLVec.vector_applicable[f32](Self.float_tanh)

  @staticmethod
  @always_inline
  fn float_tanh_prime(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return 1-(Self.float_tanh(x)**2)

  alias tanh_prime = MLVec.vector_applicable[f32](Self.float_tanh_prime)

  ########################################################################################################

  ########################################################################################################
  # defines leakyrelu activation function
  @staticmethod
  @always_inline
  fn float_leakyrelu(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return x if x > 0 else a*x

  alias leakyrelu = MLVec.vector_applicable[f32](Self.float_leakyrelu)

  @staticmethod
  @always_inline
  fn float_leakyrelu_prime(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return 1 if x > 0 else a

  alias leakyrelu_prime = MLVec.vector_applicable[f32](Self.float_leakyrelu_prime)

  ########################################################################################################

  ########################################################################################################
  # defines elu activation function
  @staticmethod
  @always_inline
  fn float_elu(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return x if x > 0 else a*(E**x-1)

  alias elu = MLVec.vector_applicable[f32](Self.float_elu)

  @staticmethod
  @always_inline
  fn float_elu_prime(borrowed x: Float32, borrowed a: Float32 = 1) -> Float32:
    return 1 if x > 0 else a*(E**x)

  alias elu_prime = MLVec.vector_applicable[f32](Self.float_elu_prime)

  ########################################################################################################

  ########################################################################################################
  # defines sigmoid activation function
  @staticmethod
  @always_inline
  fn float_sigmoid(borrowed x: Float32,borrowed a: Float32 = 1) -> Float32:
    return 1/(1+E**(-x))

  alias sigmoid = MLVec.vector_applicable[f32](Self.float_sigmoid)

  @staticmethod
  @always_inline
  fn float_sigmoid_prime(borrowed x: Float32, a: Float32 = 1) -> Float32:
    return (1-Self.float_sigmoid(x))*Self.float_sigmoid(x)

  alias sigmoid_prime = MLVec.vector_applicable[f32](Self.float_sigmoid_prime)

  ########################################################################################################

  @staticmethod
  @always_inline
  fn evaluate(activation: Int16, x: MLVec, a: Float32 = 1) raises -> MLVec:
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

  @staticmethod
  @always_inline
  fn prime_evaluate(activation: Int16, x: MLVec, a: Float32 = 1) raises -> MLVec:
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
