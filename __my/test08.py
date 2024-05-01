import steps.step08 as s
import numpy as np

def square(x):
    f = s.Square()
    return f(x)

def exp(x):
    f = s.Exp()
    return f(x)

x = s.Variable(np.array(0.5))
# a = square(x)
# b = exp(a)
# y = square(b)
y = square(exp(square(x)))

y.grad = np.array(1.0)
y.backward()
print(x.grad)