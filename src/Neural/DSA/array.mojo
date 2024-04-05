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
      self.data[i] = data[i]

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
    return self.data[index]

  fn __setitem__(inout self, owned index: Int, val: SIMD[type, 1]) raises:
    if index < (-self.length): raise Error("Index out of bounds")
    if index < 0: index = self.length + index
    if index > self.length-1: raise Error("Index out of bounds")
    self.data[index] = val

  fn __iter__(self) -> Self:
    return self

  fn __next__(inout self) raises -> SIMD[type, 1]:
    if self._current >= self.length: raise Error("End of array")
    self._current += 1
    return self.data[self._current]

  fn append(inout self, owned data: SIMD[type, 1]) raises:
    var newptr = DTypePointer[type]().alloc(self.length + 1)

    for i in range(self.length):
      newptr[i] = self[i]

    newptr[self.length] = data
    self.data = newptr

    self.length += 1

  fn sum(self) -> SIMD[type, 1]:
    var val: SIMD[type, 1] = 0
    for i in range(self.length):
      val += self.data[i]
    return val

struct Array[type: AnyRegType]:

  var data: Pointer[type]
  var length: Int
  var _current: Int

  fn __init__(inout self, *data: type):
    self.data = Pointer[type]().alloc(len(data))
    self.length = len(data)
    self._current = 0

    for i in range(len(data)):
      self.data.store[Int](i, data[i])

  fn __init__(inout self, length: Int):
    self.length = length
    self.data = Pointer[type]().alloc(length)
    self._current = 0

  fn __init__(inout self, data: Pointer[type], len: Int):
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

  fn __getitem__(self, owned index: Int) raises -> type:
    if index < (-self.length): raise Error("Index out of bounds")
    if index < 0: index = self.length + index
    if index > self.length-1: raise Error("Index out of bounds")
    return self.data.load[Int](index)

  fn __setitem__(inout self, owned index: Int, val: type) raises:
    if index < (-self.length): raise Error("Index out of bounds")
    if index < 0: index = self.length + index
    if index > self.length-1: raise Error("Index out of bounds")
    return self.data.store[Int](index, val)

  fn __iter__(self) -> Self:
    return self

  fn __next__(inout self) raises -> type:
    if self._current >= self.length: raise Error("End of array")
    self._current += 1
    return self.data.load[Int](self._current)

  fn append(inout self, owned data: type) raises:
    var newptr = Pointer[type]().alloc(self.length + 1)

    for i in range(self.length):
      newptr.store[Int](i, self[i])

    newptr.store[Int](self.length, data)
    self.data = newptr

    self.length += 1

