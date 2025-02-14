from tinytorch.core import Variable # 导入类或者函数
from tinytorch.core import Function
from tinytorch.core import as_array
from tinytorch.core import as_variable
from tinytorch.core import setup_variable
from tinytorch.core import Config
import tinytorch.functions # 导入模块
import tinytorch.utils
import tinytorch.layers
import tinytorch.models
import tinytorch.optimizers

setup_variable()

'''
导入顺序: 
1. 外部import tinytorch或from tinytorch import xxx或import tinytorch.xxx之后: 
2. 导入__init__.py
    2.1 从core中开始导入类和函数
        2.1.1 import tinytorch被跳过, 因为有缓存
    2.2 从functions开始导入类和函数
        2.2.1 import tinytorch, 被跳过
        2.2.2 from tinytorch import utils, 导入utils的类和函数
        2.2.3 导入 Function, Variable, as_variable, as_array 类和函数, 实际上已导入, 但方便使用
2.3 从utils开始导入类和函数, 被跳过, 已导入
'''