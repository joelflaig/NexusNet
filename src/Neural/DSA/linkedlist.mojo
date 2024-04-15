from memory.unsafe import Pointer

trait Node:
  pass

trait LLTrait:
  pass

struct LLNode[T: Copyable](Node):

  var data: T
  var next: Pointer[Self]

  fn __init__(inout self):
    self.data = self.data
    self.next = Pointer[Self]()

  fn __init__(inout self, data: T):
    self.data = data
    self.next = Pointer[Self]()

  fn __copyinit__(inout self, other: Self):
    self.data = other.data
    self.next = other.next

  fn point(inout self, node: Pointer[Self]):
    self.next = node

struct DLLNode[T: Copyable](Node):

  var data: T
  var next: Pointer[Self]
  var prev: Pointer[Self]

struct LinkedList[T: Copyable](LLTrait, Sized):

  var head: LLNode[T]
  var end: LLNode[T]
  var length: Int
  var _current: LLNode[T]

  fn __init__(inout self):
    self.head = LLNode[T]()
    self.end = LLNode[T]()
    self.length = 0
    self._current = LLNode[T]()

  fn __len__(self) -> Int:
    return self.length

  fn insert_front(inout self, val: T):
    var new_node = LLNode[T](val)
    new_node.point(Pointer[LLNode[T]]().address_of(self.head))
    self.head = Pointer[LLNode[T]]().address_of(new_node)

  fn append(inout self, val: T):
    var new_node = LLNode[T](val)
    self.end.point(Pointer[LLNode[T]]().address_of(new_node))
    self.end = new_node

  fn insert(inout self, val: T):
    '''Inserts `val` at the current position.'''
    var new_node = LLNode[T](val)
    new_node.point(self._current.next)
    self._current.point(Pointer[LLNode[T]]().address_of((new_node)))

  fn next(inout self):
    self._current = self._current.next

  fn goto_head(inout self):
    self._current = self.head

  fn goto_end(inout self):
    self._current = self.end

