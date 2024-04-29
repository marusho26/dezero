import steps.step07 as s
import numpy as np

A = s.Square()
B = s.Exp()
C = s.Square()

x = s.Variable(np.array(10))
a = A(x)
b = B(a)
y = C(b)

# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad = B.backward(b.grad)
# x.grad = A.backward(a.grad)
# print(f"x.data:{x.data}/x.grad{x.grad}")

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x