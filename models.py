from tinytorch import layers as L
from tinytorch import utils 
from tinytorch import functions as F
import tinytorch.cuda as cuda
import numpy as np

class Model(L.Layer):
    def plot(self, *inputs, filename = 'model'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, output_type='png', filename = filename)

class MLP(Model):
    def __init__(self, fc_output_sizes, activation = F.sigmoid):
        super().__init__()
        self.activation = activation 
        self.layers = [] 

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer) # 加入模型的参数中
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
    
class RNN(Model):
    def __init__(self, hidden_size, num_layer = 1, h_ini = None):
        super().__init__()
        self.layers = []
        self.h_ini = h_ini
        for i in range(num_layer):
            layer = L.RNNCell(hidden_size)
            setattr(self, 'l' + str(i), layer) # 加入模型的参数中, 方便获取
            self.layers.append(layer)   

    def forward(self, x):
        # x: N x Seq x Feature
        h_t_minus_1 = self.h_ini
        N, Seq, Feature = x.shape 
        for i in range(Seq):
            h_t = None
            x_t = x[:, i, :].reshape(N, Feature)
            for layer in self.layers:
                if h_t is None:
                    h_t = layer(x_t, h_t_minus_1)
                else:
                    h_t = layer(h_t, h_t_minus_1)
            h_t_minus_1 = h_t
        return h_t

class LSTM(Model):
    def __init__(self, hidden_size, num_layer = 1, h_ini = None, c_ini = None):
        super().__init__()
        self.layers = []
        self.h_ini = h_ini
        self.c_ini = c_ini
        for i in range(num_layer):
            layer = L.LSTMCell(hidden_size)
            setattr(self, 'l' + str(i), layer) # 加入模型的参数中
            self.layers.append(layer)   

    def forward(self, x):
        # x: N x Seq x Feature
        h_t_minus_1 = self.h_ini
        c_t_minus_1 = self.c_ini
        N, Seq, Feature = x.shape 
        for i in range(Seq):
            h_t = None
            c_t = None
            x_t = x[:, i, :].reshape(N, Feature)
            for layer in self.layers:
                if h_t is None:
                    h_t, c_t = layer(x_t, h_t_minus_1, c_t_minus_1)
                else:
                    h_t, c_t = layer(h_t, h_t_minus_1, c_t_minus_1)

            h_t_minus_1, c_t_minus_1 = h_t, c_t

        return h_t

class GRU(Model):
    def __init__(self, hidden_size, num_layer = 1, h_ini = None):
        super().__init__()
        self.layers = []
        self.h_ini = h_ini
        for i in range(num_layer):
            layer = L.GRUCell(hidden_size)
            setattr(self, 'l' + str(i), layer) # 加入模型的参数中, 方便获取
            self.layers.append(layer)   

    def forward(self, x):
        # x: N x Seq x Feature
        h_t_minus_1 = self.h_ini
        N, Seq, Feature = x.shape 
        for i in range(Seq):
            h_t = None
            x_t = x[:, i, :].reshape(N, Feature)
            for layer in self.layers:
                if h_t is None:
                    h_t = layer(x_t, h_t_minus_1)
                else:
                    h_t = layer(h_t, h_t_minus_1)
            h_t_minus_1 = h_t
        return h_t

class Transformer(Model):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = L.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = L.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = L.PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = []
        self.decoder_layers = []
        for i in range(num_layers):
            layer = L.EncoderLayer(d_model, num_heads, d_ff, dropout)
            setattr(self, 'encoder_layer' + str(i), layer) # 加入模型的参数中, 方便获取
            self.encoder_layers.append(layer)   

        for j in range(num_layers):
            layer = L.DecoderLayer(d_model, num_heads, d_ff, dropout)
            setattr(self, 'decoder_layer' + str(j), layer) # 加入模型的参数中, 方便获取
            self.decoder_layers.append(layer)   

        self.fc = L.Linear(d_model, tgt_vocab_size)
        self.dropout = dropout

    def generate_mask(self, src, tgt):
        xp = cuda.get_array_module(src)
        src_mask = xp.expand_dims((src != 0), axis = (1, 3)) # id为0是padding
        tgt_mask = xp.expand_dims((tgt != 0), axis = (1, 3))
        seq_length = tgt.shape[1]
        # no peek: 不要偷看
        # nopeek_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1))
        nopeek_mask = xp.ones((seq_length, seq_length))
        nopeek_mask[xp.triu_indices(seq_length, k=1)] = 0
        nopeek_mask = nopeek_mask[np.newaxis, :, :].astype("bool")

        tgt_mask = tgt_mask & nopeek_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # 训练的过程中是有tgt的!
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = F.dropout(self.positional_encoding(self.encoder_embedding(src)), drop_out_ratio=self.dropout)
        tgt_embedded = F.dropout(self.positional_encoding(self.decoder_embedding(tgt)),  drop_out_ratio=self.dropout)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output