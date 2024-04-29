import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

x = Variable(np.array(10))
print(x.data)