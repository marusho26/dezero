import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        # print('funcs_1:',funcs)
        while funcs:
            # print('funcs_2:',funcs)
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
                # print('funcs_3',funcs)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # foward関数は、変化させたい＆その関数の呼び出しはクラス内で行われる
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx