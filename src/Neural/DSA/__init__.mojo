"""Module providing Datastructures and algorithms."""

from .tensorfuncs import *
from .array import *
from tensor import Tensor

alias bool = DType.bool
'''
This is an alias for your(and my) convienience.

It represents a boolean data type.
'''

alias si8 = DType.int8
'''
This is an alias for your(and my) convienience.

It represents a signed integer data type whose bitwidth is 8.
'''

alias si16 = DType.int16
'''
This is an alias for your(and my) convienience.

It represents a signed integer data type whose bitwidth is 16.
'''

alias si32 = DType.int32
'''
This is an alias for your(and my) convienience.

It represents a signed integer data type whose bitwidth is 32.
'''

alias si64 = DType.int64
'''
This is an alias for your(and my) convienience.

It represents a signed integer data type whose bitwidth is 64.
'''

alias ui8 = DType.uint8
'''
This is an alias for your(and my) convienience.

It represents an unsigned integer data type whose bitwidth is 8.
'''

alias ui16 = DType.uint16
'''
This is an alias for your(and my) convienience.

It represents an unsigned integer data type whose bitwidth is 16.
'''

alias ui32 = DType.uint32
'''
This is an alias for your(and my) convienience.

It represents an unsigned integer data type whose bitwidth is 32.
'''

alias ui64 = DType.uint64
'''
This is an alias for your(and my) convienience.

It represents an unsigned integer data type whose bitwidth is 64.
'''

alias f16 = DType.float16
'''
This is an alias for your(and my) convienience.

It represents a float data type whose bitwidth is 16.
'''

alias f32 = DType.float32
'''
This is an alias for your(and my) convienience.

It represents a float data type whose bitwidth is 32.
'''

alias f64 = DType.float64
'''
This is an alias for your(and my) convienience.

It represents a float data type whose bitwidth is 64.
'''

alias tf32 = DType.tensor_float32
'''
This is an alias for your(and my) convienience.

It represents a special float format supported by NVIDIA Tensor Cores.

This type is only available on NVIDIA GPUs.
'''

alias MLT = Tensor[f32]
'''
`Tensor` containing `DType.float32` values useful for ML applications.
'''

alias GPUT = Tensor[tf32]
'''
`Tensor` containing values of Mojo's special `DType.tensor_float32` type.

Note that this requires a NVIDIA GPU.
'''
