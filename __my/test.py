import steps.step07 as s
import numpy as np

A = s.Square()
B = s.Exp()
C = s.Square()

x = s.Variable(np.array(10))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(f"x.data:{x.data}/x.grad{x.grad}")