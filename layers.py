from tinytorch.core import Parameter, Variable
import tinytorch.functions as F
import tinytorch.cuda as cuda
import weakref
import numpy as np
import math

'''
Layer类是持有模型所有参数的类. 所有待学习的参数都必须在Layer中. 
Linear类是非常基本的操作单元. 
1. Layer在forward中利用Function建立计算图. Function是反向传播的基本单元.
2. Layer通过非常巧妙的结构设计, 使得访问以Layer及其子类构成的模型的参数非常方便.
3. 所有的模型都要继承Model, 也就是Layer的子类
'''
# -------------------------------------------------------------------------------
# 基类
# ------------------------------------------------------------------------------

class Layer:
    def __init__(self) -> None:
        self._params = set()
    
    # 子类使用"."设置属性值的时候会被调用.
    def __setattr__(self, name: str, value):
        if isinstance(value, (Parameter, Layer)): # Layer中可以存储Layer
            self._params.add(name) # 保留变量的名字. 这是实实在在的属性名字.
        super().__setattr__(name, value) # 不使用self, 防止递归

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        # 弱引用持有
        self.inputs = [weakref.ref(x) for x in inputs if x is not None]
        self.outputs = [weakref.ref(y) for y in outputs if y is not None]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError 
    
    # 参数的访问接口，是一个iterator
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params() # 这实际上是一个递归
            else:
                yield obj 
    
    def clear_grads(self):
        for param in self.params():
            param.clear_grad()
    
    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
    
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

# ---------------------------------------------------------------------------------------
# 线性变换层
# ---------------------------------------------------------------------------------------

class Linear(Layer):
    def __init__(self, out_size, nobias = False, dtype = np.float32, in_size = None):
        super().__init__() # 召唤出一个_params
        self.in_size = in_size 
        self.out_size = out_size 
        self.dtype = dtype 

        # 当声明一个Parameter, 或者Layer的子类, 会被加入参数中.
        self.W = Parameter(None, name = "W") # 延迟创建, 可以自动输入判断维度.

        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None 
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name = 'b') # 维度不是冗余的.
        
    def _init_W(self, xp = np):
        I, O = self.in_size, self.out_size 
        '''
        初始化依据于某论文. 变换后方差为 1/I 
        变换为 XW = Y. 
        '''
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I) 
        self.W.data = W_data 
    
    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[-1]
            xp = cuda.get_array_module(x)
            self._init_W(xp = xp) 
        
        y = F.linear(x, self.W, self.b) 
        return y

# ------------------------------------------------------------------------------
# Normalization Layer
# -----------------------------------------------------------------------------
'''
Applies Batch Normalization over a 2D or 3D input.
'''
# parameter用numpy数组初始化没事的，因为to_gpu方法可以将Parameter悉数搬运到gpu
class BatchNorm1D(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = Parameter(np.array(1.0))
        self.beta = Parameter(np.array(0.0))

        self.batch_norm = F.BatchNorm(mean=0, var=1, decay=0.9, eps=2e-5)
    def forward(self, x):
        return self.batch_norm(x, self.gamma, self.beta)
    
class LayerNorm(Layer):
    def __init__(self, dimension) -> None:
        super().__init__()
        if isinstance(dimension, int):
            self.dimension = -1 
        if isinstance(dimension, (tuple, list)):
            self.dimension = [-(i+1) for i in range(len(dimension))]
        self.gamma = Parameter(np.array(1.0))
        self.beta = Parameter(np.array(0))

    def forward(self, x):
        return F.layernormfunction(x, self.gamma, self.beta, self.dimension)

# ----------------------------------------------------------------------------
# Recurrent Neural
# ----------------------------------------------------------------------------

class RNNCell(Layer):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.x2h = Linear(out_size = hidden_size)
        self.h2h = Linear(out_size = hidden_size)
    
    # 处理一个时间点
    def forward(self, x, h):
        # x: N x Feature
        if h is not None:
            h_new = F.tanh(self.h2h(h) + self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x))

        return h_new 
    
class LSTMCell(Layer):
    def __init__(self, hidden_size, sigma = F.tanh) -> None:
        super().__init__()
        # 设置属性, 加入Layer的参数列表中
        self.Wfx = Linear(out_size=hidden_size, nobias=True)
        self.Wfh = Linear(out_size=hidden_size)
        self.Wix = Linear(out_size=hidden_size, nobias=True)
        self.Wih = Linear(out_size=hidden_size)
        self.Wox = Linear(out_size=hidden_size, nobias=True)
        self.Woh = Linear(out_size=hidden_size)
        self.Wtransx = Linear(out_size=hidden_size, nobias=True)
        self.Wtransh = Linear(out_size=hidden_size)

        self.sigma = sigma

    def forward(self, x, h, c):
        # x: N x Feature
        if h is not None:
            f_t = self.sigma(self.Wfx(x) + self.Wfh(h))
            i_t = self.sigma(self.Wix(x) + self.Wih(h))
            o_t = self.sigma(self.Wox(x) + self.Woh(h))
            c_tilde_t = F.tanh(self.Wtransx(x) + self.Wtransh(h))

            c_t = f_t*c + i_t*c_tilde_t 
            
        else:
            f_t = self.sigma(self.Wfx(x))
            i_t = self.sigma(self.Wix(x))
            o_t = self.sigma(self.Wox(x))
            c_tilde_t = F.tanh(self.Wtransx(x))

            c_t = i_t*c_tilde_t

        h_t = o_t*F.tanh(c_t)
        return h_t, c_t
    
class GRUCell(Layer):
    def __init__(self, hidden_size, sigma = F.tanh) -> None:
        super().__init__()
        self.Wux = Linear(out_size=hidden_size, nobias=True) 
        self.Wuh = Linear(out_size=hidden_size)
        self.Wrx = Linear(out_size=hidden_size, nobias=True)
        self.Wrh = Linear(out_size=hidden_size)

        self.Whx = Linear(out_size=hidden_size, nobias=True)
        self.Whh = Linear(out_size=hidden_size)

        self.sigma = sigma 
    def forward(self, x, h):
        if h is not None:
            z_t = self.sigma(self.Wux(x) + self.Wuh(h))
            r_t = self.sigma(self.Wrx(x) + self.Wrh(h))

            hp_t_minus_1 = r_t*h
            hp = F.tanh(self.Whx(x) + self.Whh(hp_t_minus_1))
            h_t = (1-z_t)*h + z_t*hp
        else:
            z_t = self.sigma(self.Wux(x))
            r_t = self.sigma(self.Wrx(x))

            hp = F.tanh(self.Whx(x))
            h_t = z_t*hp            

        return h_t
    
# ----------------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------------

class Embedding(Layer):
    def __init__(self, vocab_size, d_model) -> None:
        super().__init__()
        self.E = Parameter(np.random.randn(vocab_size, d_model))
    def forward(self, x):
        # x: id向量
        return self.E[x]

# ---------------------------------------------------------------------------------
# Transformer 
# ---------------------------------------------------------------------------------

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        # d_model: 特征维度, 在NLP中就是embedding的维度加上position_encoding
        # num_heads: 将特征切分成多少份, 就是多少个头

        super().__init__()
        assert d_model % num_heads == 0, "特征维度必须整除头数"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头是多少维度

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_0 = Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        # 这里进来的是已经切分好的Q, K, V
        # Q: batch_size * num_heads * seq_length * d_k
        # K: batch_size * num_heads * seq_length * d_k
        # V: batch_size * num_heads * seq_length * d_k

        # 计算注意力得分
        attn_scores = F.matmul(Q, K.transpose((0,1,3,2))) 
        attn_scores /= math.sqrt(self.d_k)

        # 使用mask
        if mask is not None:
            mask[mask == 0] = 1e-9
            attn_scores = attn_scores*mask
        
        # 使用softmax将最后一个维度归一化
        attn_probs = F.softmax(attn_scores)

        # 计算拉直之前的输出
        output = F.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # 将特征维(一般是最后一维)切分
        batch_size, seq_length, d_model = x.shape
        # 注意这里有个transpose()
        return x.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose((0, 2, 1, 3))
    
    def combine_heads(self, x):
        # 这里是上一步转换了维度的, 这一步又要转换回来
        batch_size, _, seq_length, d_k = x.shape
        return x.transpose((0, 2, 1, 3)).reshape(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask = None):
        # Q: batch_size * seq_length * d_model
        # K: batch_size * seq_length * d_model
        # V: batch_size * seq_length * d_model
        # 首先进来之后先做维度保持的线性变换然后将维度进行切割
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 然后计算输出
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 最后, 将切割的维度重新组合起来
        output = self.W_0(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = Linear(out_size=d_ff)
        self.fc2 = Linear(out_size=d_model)
        self.relu = F.ReLU()

    def forward(self, x):
        # 是否是对最后一个维度进行改变的?
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(Layer):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
        pe = np.zeros((max_seq_length, d_model))

        # 一个从0到max_length的向量.
        position = np.expand_dims(np.arange(0, max_seq_length, dtype=np.float32), axis = 1)

        # 决定positional encoding的维度和特征.
        div_term = np.exp(np.arange(0, d_model, 2).astype("float32") * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = Parameter(pe)
        self.pe.require_grad = False
    def forward(self, x):
        return x + self.pe[:x.shape[1], :]
    
class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = dropout
        
    def forward(self, x, mask):
        # x: batch * seq_length * feature_length
        attn_output = self.self_attn(x, x, x, mask) # mask的作用是让padding位置注意力为0
        # attn_output: batch * seq_length * feature_length
        x = self.norm1(x + F.dropout(attn_output, drop_out_ratio=self.dropout))
        # x: batch * seq_length * feature_length
        ff_output = self.feed_forward(x)
        # x: batch * seq_length * feature_length
        x = self.norm2(x + F.dropout(ff_output, self.dropout))
        # x: batch * seq_length * feature_length
        return x
    
class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = dropout
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + F.dropout(attn_output, drop_out_ratio=self.dropout))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + F.dropout(attn_output, drop_out_ratio=self.dropout))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + F.dropout(ff_output, drop_out_ratio=self.dropout))

        return x