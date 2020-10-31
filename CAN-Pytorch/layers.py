import torch
from torch import sqrt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
import torch.sparse

latent_dim = 128
hidden_decoder_dim = 512



def weight_variable_glorot(input_dim, output_dim):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.FloatTensor(input_dim,output_dim).uniform_(-init_range,init_range)   
    # tf.random_uniform([input_dim, output_dim], minval=-init_range,maxval=init_range, dtype=tf.float32)
    return nn.Parameter(initial, requires_grad=True)


class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, dropout=0.):
        super(GraphConvolution, self).__init__()
        self.weights = weight_variable_glorot(input_dim, output_dim)
        self.dropout = dropout

    def forward(self, inputs,adj):
        x = F.dropout(inputs,self.dropout,self.training)
        x = torch.mm(x, self.weights)
        outputs = torch.mm(adj.to_dense(),x)
        return outputs


class GraphConvolutionSparse(nn.Module):
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, features_nonzero, dropout=0.):
        super(GraphConvolutionSparse, self).__init__()
        self.weights = weight_variable_glorot(input_dim, output_dim)
        self.dropout = dropout
        self.issparse = True
        self.features_nonzero = features_nonzero

    def forward(self, inputs,adj):
        x = inputs.to_dense()
        x = torch.mm(x, self.weights)
        outputs = torch.mm(adj.to_dense(), x)
        return outputs


class Dense(nn.Module):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, dropout=0., bias=True,sparse_inputs=False):
        super(Dense, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs

        self.weights = weight_variable_glorot(input_dim, output_dim)
        if self.bias:
            self.bias = Parameter(torch.zeros([output_dim], dtype=torch.float32))

    def forward(self, inputs):
        x = inputs

        if self.sparse_inputs:
            output = torch.sparse.mm(x, self.weights)
        else:
            output = torch.mm(x, self.weights)

        # bias
        output += self.bias

        return output


# class InnerProductDecoder(nn.Module):
#     """Decoder model layer for link prediction."""

#     def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
#         super(InnerProductDecoder, self).__init__(**kwargs)
#         self.dropout = dropout
#         self.act = act

#     def forward(self, inputs):
#         inputs = nn.Dropout(inputs, 1 - self.dropout)#修改
#         print("inputs.shape:", inputs)
#         x = inputs.t()####修改
#         x = torch.mm(inputs, x)
#         print("x = tf.matmul(inputs, x):", x)

#         x = x.view(-1)#######修改
#         outputs = self.act(x)
#         return outputs



class InnerDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0.):
        super(InnerDecoder, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim

    def forward(self, z_u,z_a):
        z_u = F.dropout(z_u,self.dropout,self.training)
        z_u_t = z_u.t()
        x = torch.mm(z_u, z_u_t)
        z_a_t = F.dropout(z_a, self.dropout,self.training).t()
        y = torch.mm(z_u, z_a_t)
        edge_outputs = x.flatten()
        attri_outputs = y.flatten()
        return edge_outputs, attri_outputs
