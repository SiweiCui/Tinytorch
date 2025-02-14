from tinytorch import utils
from tinytorch import Function, Variable, as_variable, as_array, Config
import numpy as np
from tinytorch import cuda

# --------------------------------------------------------
# 数学函数/激活函数
# --------------------------------------------------------
class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.exp(x)
    def backward(self, gy):
        return gy * self.outputs[0]() # weakref
def exp(x):
    return Exp()(x)

class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.log(x)
    def backward(self, gy):
        return gy / self.inputs[0]
def log(x):
    return Log()(x)

class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return 1 / (1 + xp.exp(-x))
    def backward(self, gy):
        x = self.inputs[0]
        y = 1 / (1 + exp(-x))
        return gy * y * (1 - y)
def sigmoid(x):
    return Sigmoid()(x)

class Softmax(Function):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True) # 数值更加稳定
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=-1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=-1):
    return LogSoftmax(axis)(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0)
        return y 
    def backward(self, gy):
        mask = (self.inputs[0].data > 0)
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)

class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max) 
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# 也会发生反向传播, 在x*msak/scale位置. 只是不作为一个单独的单元.
def dropout(x, drop_out_ratio = 0.5):
    x = as_variable(x)

    if Config.enable_backprop: # 训练模式
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > drop_out_ratio 
        scale = xp.array(1 - drop_out_ratio).astype(x.dtype)
        y = x*mask/scale 
        return y 
    else:
        return x


# 只针对二维矩阵
class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        '''
        gamma, beta: learnable parameters
        训练的时候就正常. 测试的时候用到训练的moving average
        '''
        xp = cuda.get_array_module(x)

        if Config.enable_backprop: # 训练阶段
            mean = x.mean(axis=1) 
            var = x.var(axis=1) 
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1. if m - 1. > 1. else 1.
            adjust = m / s  # unbiased estimation
            # moving average
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else: # 测试阶段使用训练阶段的moving average
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std 

        y = gamma * xc + beta


        return y

    def backward(self, gy):
        gy_ndim = gy.ndim

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        return gx, ggamma, gbeta


def batch_norm(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


class LayerNormFunction(Function):
    def __init__(self, dimension):
        self.dimension = dimension

    def forward(self, x, gamma, beta):
        # gamma: N x Seq x 1
        # beta: N x Seq x 1
        eps = 2e-5
        xp = cuda.get_array_module(x)

        mean = xp.expand_dims(x.mean(axis=self.dimension), axis=-1)
        var = x.var(axis=self.dimension) 
        self.inv_std = xp.expand_dims(1 / xp.sqrt(var + eps), axis=-1)
        xc = (x - mean) * self.inv_std

        y = gamma * xc + beta
        return y 
    def backward(self, gy):
        x, gamma, beta = self.inputs
        xp = cuda.get_array_module(x)
        gbeta = gy.sum(axis = self.dimension)
        mean = xp.expand_dims(x.mean(axis=self.dimension).data, axis=-1)
        xc = (x - mean) * self.inv_std 
        ggamma = (gy * xc).sum(axis = self.dimension)
        gx = gy * gamma * self.inv_std

        return gx, gbeta, ggamma

def layernormfunction(x, gamma, beta, dimension):
    return LayerNormFunction(dimension)(x, gamma, beta)

    

# -----------------------------------------------------------
# 矩阵计算
# -----------------------------------------------------------
class MatMul(Function):
    def forward(self, x, M):
        if len(x.shape)<=2 and len(M.shape)<=2:
            y = x.dot(M)
        else:
            xp = cuda.get_array_module(x)
            y = xp.matmul(x, M)
        return y 
    def backward(self, gy):
        # gy 总是与 y 同形状. 而且机器学习的损失函数总为标量. 所以分析有效.
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(x.T, gy)
        return gx, gw 
def matmul(x, W):
    return MatMul()(x, W)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W) # 矩阵乘法
        if b is not None:
            y += b # 可能会发生广播. 如y: (100, 4), b: (4, ). 如果发生了广播, b每个元素都做了100次贡献.
        return y
    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)

        shape = [d for d in range(len(x.shape))]
        shape[-1], shape[-2] = shape[-2], shape[-1]
        gW = matmul(x.transpose(shape), gy)
        return gx, gW, gb
def linear(x, W, b=None):
    return Linear()(x, W, b)

# -------------------------------------------------------------
# 损失函数
# ------------------------------------------------------------
class MeanSquareError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        loss = (diff ** 2).sum()/len(diff)
        return loss 
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * 2 * diff / len(diff)
        gx1 = - gy * 2 * diff / len(diff) 
        return gx0, gx1
def mean_squared_error(pre, true):
    return MeanSquareError()(pre, true)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=-1) # Nx1
        log_p = x - log_z # 概率向量, NxC
        xp = cuda.get_array_module(x)
        log_p = log_p[xp.arange(N), t.ravel()] # 一个N维向量
        y = -log_p.sum() / xp.float32(N)
        return y 

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y # 实际上, 反向传播应该返回两个, 但是zip里面如果自动填充None, 那也就没事了.


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y



# --------------------------------------------------------
# 改变形状的函数
# 1.
# reshape, transpose, sum以及广播, 要用element-wise角度去看待. 看每个输入自变量的变化和在y中的贡献.
# 而不是多维y与多维x的对应关系. 摆脱传统矩阵求导的框架. 
# 2.
# backward中使用的函数, 如果不是为了高阶导数, 就仅仅是为了用他们的forward而已. 不要晕了.
# --------------------------------------------------------
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape 
    def forward(self, x): # 仍然是对ndarray的操作
        self.x_shape = x.shape 
        y = x.reshape(self.shape)
        return y 
    def backward(self, gy):
        return reshape(gy, self.x_shape)    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)



class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        self.x_shape = x.shape 
        y = x.sum(axis = self.axis, keepdims = self.keepdims) 
        return y 
    def backward(self, gy):
        # 将gy进行一个转换
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        # 转换后, gy与gx的shape维度个数一致. 尽管有些地方是1.
        gx = broadcast_to(gy, self.x_shape)
        return gx 
def sum(x, axis = None, keepdims = False):
    return Sum(axis, keepdims)(x)

class Mean(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        self.x_shape = x.shape 
        y = x.mean(axis = self.axis, keepdims = self.keepdims) 
        return y 
    def backward(self, gy):
        n = 1
        for i in self.axis:
            n *= self.x_shape[i]
        # 将gy进行一个转换
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims) / n
        # 转换后, gy与gx的shape维度个数一致. 尽管有些地方是1.
        gx = broadcast_to(gy, self.x_shape)
        return gx 
def mean(x, axis = None, keepdims = False):
    return Mean(axis, keepdims)(x)



class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape 
    def forward(self, x):
        xp = cuda.get_array_module(x)
        self.x_shape = x.shape
        y = xp.broadcast_to(x, self.shape)
        return y 
    def backward(self, gy):
        # 因为broadcast往往意味着元素被重复使用, 所以要按照某个方向把元素的梯度相加
        gx = sum_to(gy, self.x_shape)
        return gx 
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y 
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape) # 加法, 平摊梯度
        return gx 
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
'''
如果被sumto和广播的形状变换困扰, 就举几个例子试一试啊. 所有例子都通过了就是对的.
干想的话太抽象了.
'''

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        xp.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)



'''
y: 概率向量: NxC
t: 标签向量: Nx1
'''
def accuracy(y, t):
    y, t = as_variable(y), as_variable(t) 

    pred = y.data.argmax(axis = 1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean() 

    return Variable(as_array(acc))




