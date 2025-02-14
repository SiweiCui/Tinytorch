import numpy as np
import heapq 
import weakref # 引入弱引用, 解决循环引用问题
from typing import *
from types import ModuleType
import heapq 
import contextlib
import tinytorch
'''
此处tinytorch不会导致循环导入, 因为在外部导入tinytorch时,python会缓存一个tinytorch, 然后调用__init__.py, 
__init__.py中要导入core, 随即python开始导入本文件, 运行到这里时, 系统认为tinytorch已经被导入, 所以会略过. 

但是你要是在这里导入functions模块就不可以, 因为functions模块需要Function类. 而此类还没有导入.
'''

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


# 将numpy中的标量转换成array
def as_array(x: Union[np.ndarray, cupy.ndarray], array_module: ModuleType = np):
    if np.isscalar(x):
        return array_module.array(x)
    return x

# np.array转换成Variable
def as_variable(obj: Union[np.ndarray, cupy.ndarray]):
    if isinstance(obj, Variable):
        return obj 
    return Variable(obj)



class Variable:
    # 获得比ndarray高的优先级, 处理左项为ndarray的情况. 优先调用本类的方法.
    __array_priority__ = 200
    def __init__(self, data:np.ndarray, name: str = None ) -> None:
        if data is not None and not isinstance(data, array_types):
            raise TypeError('{} is not supported'.format(type(data)))
        self.data = data 
        self.name = name 
        self.grad = None 
        self.creator = None 
        self.generation = 0 
        self.require_grad = True
    def set_creator(self, func) -> None:
        self.creator = func 
        self.generation = func.generation + 1 
    def backward(self, retain_grad = False, create_graph = False) -> None:
        '''
        retain_grad: 当前变量的下一级变量的梯度是否需要保留
        create_graph: 在backward中使用了Variable, 需要高阶导数的场景时, 会为每个grad创建计算图. 否则不需要为grad创建计算图. 
        节省内存
        '''
        if self.grad is None: 
            xp = tinytorch.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))
        funcs = [self.creator]
        heapq.heapify(funcs)
        seen = set()
        seen.add(id(self.creator))
        # AD算法核心
        while funcs:
            func = heapq.heappop(funcs)
            gys = [output().grad for output in func.outputs] # 收集所有输出, 实际上多输出非常罕见
            with Config.using_config('enable_backprop', create_graph):
                gxs = func.backward(*gys)
                if not isinstance(gxs, tuple): gxs = (gxs, )
                for x, gx in zip(func.inputs, gxs):
                    if x.require_grad:
                        if x.grad is None:
                            x.grad = gx
                        else:
                            x.grad = x.grad + gx             
                    if x.creator is not None and id(x.creator) not in seen:
                            heapq.heappush(funcs, x.creator)
                            seen.add(id(x.creator))
            # 清除输出端的梯度
            if not retain_grad:
                for output in func.outputs:
                    output().grad = None # output是弱引用
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return tinytorch.functions.reshape(self, shape)
    def sum(self, axis = None, keepdims = False):
        return tinytorch.functions.sum(self, axis, keepdims)
    def mean(self, axis = None, keepdims = False):
        return tinytorch.functions.mean(self, axis, keepdims)
    def clear_grad(self) -> None:
        self.grad = None 
    def to_cpu(self):
        if self.data is not None:
            self.data = tinytorch.cuda.as_numpy(self.data)
    def to_gpu(self):
        if self.data is not None:
            self.data = tinytorch.cuda.as_cupy(self.data)
    def transpose(self, axes = None):
        return tinytorch.functions.transpose(self.data, axes=axes)

    @property
    def shape(self):
        return self.data.shape 
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size 
    @property 
    def dtype(self):
        return self.data.dtype 
    @property
    def T(self):
        return tinytorch.functions.transpose(self)
    def __len__(self) -> int:
        return len(self.data)
    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'
    
'''
与Variable以示区别
'''
class Parameter(Variable):
    pass

class Config:
    enable_backprop = True 

    # 用于测试集
    @staticmethod
    @contextlib.contextmanager
    def no_grad(): # 在Function的__call__中, 不会建立连接
        setattr(Config, "enable_backprop", False)
        try:
            yield 
        finally:
            setattr(Config, "enable_backprop", True) 

    # 用于高阶导数图的建立
    @staticmethod
    @contextlib.contextmanager
    def using_config(name, value):
        old_value = getattr(Config, name)
        setattr(Config, name, value)
        try:
            yield 
        finally:
            setattr(Config, name, old_value)


'''
__call__: Variable / Tuple[Variable] -> Variable / Tuple[Variable] 所有连接在call中建立
forward: np.ndarray -> np.ndarray
backward: Variable / Tuple[Variable] -> Variable / Tuple[Variable]
'''
class Function:
    def __call__(self, *inputs):
        # 如果inputs是ndarray, 要先进行转化. 使得Function支持ndarray
        inputs = [as_variable(input) for input in inputs]
        # calculation
        x = [input.data for input in inputs]
        y = self.forward(*x)
        if not isinstance(y, tuple):
            y = (y, )
        outputs = [Variable(as_array(array_output)) for array_output in y]
        # connection
        if Config.enable_backprop:
            self.inputs = inputs 
            self.outputs = [weakref.ref(output) for output in outputs]
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)
        return outputs if len(outputs) > 1 else outputs[0] # 多个输出变量的例子实际上很少见
    def forward(self):
        raise NotImplementedError 
    def backward(self):
        raise NotImplementedError
    def __lt__(self, other) -> bool:
        # 定义小于操作，heapq 会使用这个来比较对象
        return self.generation > other.generation # 我们generation大的应该在前面. 而headpq是小顶堆.
    def __eq__(self, other) -> bool:
        # 定义等于操作，用于比较两个对象是否相等
        if isinstance(other, Function):
            return self.generation == other.generation
        return False

class Add(Function):
    def forward(self, *x):
        # 支持广播
        x0, x1 = x
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1
    def backward(self, gy):
        gx0, gx1 = gy, gy 
        if self.x0_shape != self.x1_shape:
            gx0 = tinytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = tinytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Mul(Function):
    def forward(self, *x):
        # 支持广播
        x0, x1 = x
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 * x1 
    def backward(self, gy): 
        gx0, gx1 = gy*self.inputs[1], gy*self.inputs[0]
        if self.x0_shape != self.x1_shape:
            gx0 = tinytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = tinytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Neg(Function):
    def forward(self, x):
        return -x 
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        # 支持广播
        y = x0 - x1 
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return y 
    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = tinytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = tinytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Div(Function):
    def forward(self, x0, x1):
        # 支持广播
        y = x0 / x1 
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return y 
    def backward(self, gy):
        x0 = self.inputs[0]
        x1 = self.inputs[1] 
        gx0, gx1 = gy/x1, gy*(-x0/x1**2)
        if self.x0_shape != self.x1_shape:
            gx0 = tinytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = tinytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Pow(Function):
    def __init__(self, c: int) -> None:
        self.c = c 
    def forward(self, x):
        y = x ** self.c 
        return y 
    def backward(self, gy):
        x = self.inputs[0] 
        c = self.c 
        return c * x ** (c-1) * gy

def add(x1, x2):
    # 支持float和int
    x2 = as_array(x2, tinytorch.cuda.get_array_module(x1.data))
    return Add()(x1, x2)
def mul(x1, x2):
    # 支持float和int
    x2 = as_array(x2, tinytorch.cuda.get_array_module(x1.data))
    return Mul()(x1, x2)
def neg(x):
    return Neg()(x)
def sub(x0, x1):
    x1 = as_array(x1, tinytorch.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)
def rsub(x0, x1):
    x1 = as_array(x1, tinytorch.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)
def div(x0, x1):
    x1 = as_array(x1, tinytorch.cuda.get_array_module(x0.data))
    return Div()(x0, x1)
def rdiv(x0, x1):
    x1 = as_array(x1, tinytorch.cuda.get_array_module(x0.data))
    return Div()(x1, x0)
def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    import tinytorch.functions as F
    Variable.__add__ = add 
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div 
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = F.get_item

    