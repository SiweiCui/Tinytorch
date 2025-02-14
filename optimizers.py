import numpy as np
class Optimizer:
    def __init__(self) -> None:
        self.target = None 
        self.hooks = [] # 钩子
    
    def setup(self, target):
        self.target = target 
        return self 

    def update(self):
        # 收集反向传播后梯度非None的参数
        params = [p for p in self.target.params() if p.grad is not None]
        # 预处理(梯度裁剪等)
        for f in self.hooks:
            f(params)
        # 更新参数
        for param in params:
            if param.require_grad:
                self.update_one(param) 
        
    def update_one(self, param):
        raise NotImplementedError 
    
    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):
    def __init__(self, lr = 1e-2):
        super().__init__() # 召唤target和hooks
        self.lr = lr 
    
    def update_one(self, param):
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr = 0.01, momentum = 0.9):
        super.__init__()
        self.lr = lr 
        self.momentum = momentum 
        self.vs = {}
    
    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        v = self.vs[v_key]
        v *= self.momentum 
        v -= self.lr * param.grad.data 

        param.data += v 

class Adam(Optimizer):
    def __init__(self, lr = 0.01, beta1 = 0.9, beta2 = 0.9, epsilon = 1e-3) -> None:
        super().__init__()
        self.lr = lr 
        self.beta1 = beta1 
        self.beta2 = beta2
        self.epsilon = epsilon
        self.memory = {} 
    
    def update_one(self, param):
        key = id(param) 
        if key not in self.memory:
            self.memory[key] = [np.zeros_like(param.data), np.zeros_like(param.data)] # M, G

        Mhat = (self.beta1 * self.memory[key][0] + (1-self.beta1) * param.grad.data) / (1-self.beta1**2)
        Ghat = (self.beta2 * self.memory[key][1] + (1-self.beta2) * param.grad.data**2) / (1-self.beta2**2)

        param.data -= self.lr / (Ghat + self.epsilon) * Mhat
