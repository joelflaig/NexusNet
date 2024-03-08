'''
Contains `Array` struct.
'''

struct DTypeArray[type: DType](Sized, Copyable, Movable):
  '''
  A struct implementing an array datastructure capable of holding `DType` values.
  '''

  var data: DTypePointer[type]
  var length: Int
  var _current: Int

  fn __init__(inout self, length: Int):
    self.length = length
    self.data = DTypePointer[type]().alloc(length)
    self._current = 0

  fn __init__(inout self, *data: SIMD[type, 1]):
    self.length = len(data)
    self.data = DTypePointer[type]().alloc(self.length)
    self._current = 0

    for i in range(self.length):
      self.data.simd_store(i, data[i])

  fn __init__(inout self, data: DTypePointer[type], len: Int):
    self.data = data
    self.length = len
    self._current = 0

  fn __del__(owned self):
    self.data.free()
    self.data.__del__()
    self.length.__del__()

  fn __copyinit__(inout self, borrowed other: Self):
    self.data = other.data
    self.length = other.length
    self._current = other._current

  fn __moveinit__(inout self, owned other: Self):
    self.data.__moveinit__(other.data)
    self.length.__moveinit__(other.length)
    self._current.__moveinit__(other._current)

  fn __len__(self) -> Int:
    return self.length

  fn __getitem__(self, owned index: Int) raises -> SIMD[type, 1]:
    if index < (-self.length): raise Error("Index out of bounds")
    if index < 0: index = self.length + index
    if index > self.length-1: raise Error("Index out of bounds")
    return self.data.simd_load[1, Int](index)

  fn __setitem__(inout self, owned index: Int, val: SIMD[type, 1]) raises:
    if index < (-self.length): raise Error("Index out of bounds")
    if index < 0: index = self.length + index
    if index > self.length-1: raise Error("Index out of bounds")
    return self.data.simd_store[1](index, val)

  fn __iter__(self) -> Self:
    return self

  fn __next__(inout self) raises -> SIMD[type, 1]:
    if self._current >= self.length: raise Error("End of array")
    self._current += 1
    return self.data.simd_load[1, Int](self._current)
  
  fn append(inout self, owned data: SIMD[type, 1]):
    self.data.simd_store[1](self.length, data)
    self.length += 1

  fn sum(self) -> SIMD[type, 1]:
    var val: SIMD[type, 1] = 0
    for i in range(self.length):
      val += self.data.simd_load[1, Int](i)
    return val
