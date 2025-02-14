import os 
import subprocess
import numpy as np
from tinytorch import cuda

# 生成变量的dot语言
def _dot_var(v, verbose = False):
    dot_var = '{} [label = "{}", color = orange, style = filled]\n'

    name = ' ' if v.name is None else v.name 
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.data.shape) + " " + str(v.data.dtype) 
    return dot_var.format(id(v), name) 

# 生成函数的dot语言
def _dot_func(f):
    dot_func = '{} [label = "{}", color = lightblue, style = filled, shape = box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y())) # y是weakref
    return txt 

# 为某个y生成完整的dot语言
def get_dot_graph(output, verbose = True):
    txt = _dot_var(output)
    funcs = [output.creator]
    seen = set([id(output.creator)])  # 防止在dot中出现重复的结点以及重复的函数

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func) # 函数的dot语言包括 函数结点本身 及 其与输入输出的连接
        for x in func.inputs:
            if id(x) not in seen:
                txt += _dot_var(x, verbose) # 结点本身
                seen.add(id(x))

            if x.creator is not None:
                funcs.append(x.creator)
                seen.add(id(x.creator))
    return 'digraph g {\n' + txt + '}'

# 自动运行脚本
def plot_dot_graph(output, verbose = True, output_type = 'png', filename = "graph"):
    dot_graph = get_dot_graph(output, verbose)

    # 将dot数据保存至文件
    with open("temp_dot.dot", "w") as f:
        f.write(dot_graph)
    
    # 调用dot命令
    cmd = 'dot {} -T {} -o {}'.format("./temp_dot.dot", output_type, "./" + filename + "." + output_type)
    print(cmd)
    subprocess.run(cmd, shell = True)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        '''
        ndim == 0: x是标量
        tupled_axis is None: 对所有元素求和, 不指明方向
        若以上三件事都不成立, 此时的y比x少len(tupled_axis)个维度.
        我们要将y还原到跟x一样的维度
        '''
        # 如果指定的轴为负数, 那么就是从倒着开始数的, 需要加dim转换到正常的轴
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        
        shape = list(gy.shape)
        # 将缺失的轴加上去. 缺失的轴就是actual_axis中的值代表的轴.
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.
    Args:
        x (ndarray): Input array.
        shape:
    Returns:
        ndarray: Output array of the shape.
    """

    ndim = len(shape)
    '''
    如果输入维度数量大于输出维度数量, 那么说明需要squeeze挤掉一些维度. 
    这里默认多出来的维度总是前几个维度.
    '''
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    '''
    在前几个维度的基础上, 输出维度为1的轴, 就是需要求和的轴. 这里求出了shape中的1在输入x中的对应轴
    '''
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1]) 
    y = x.sum(lead_axis + axis, keepdims=True)

    # 将前几个维度挤掉.
    if lead > 0:
        y = y.squeeze(lead_axis)

    # 总而言之, 这函数就是, 目标shape有1的地方, 在x里面抹掉, 没有1, 抹掉x前面多出来的轴.
    
    return y

def logsumexp(x, axis=1):
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m