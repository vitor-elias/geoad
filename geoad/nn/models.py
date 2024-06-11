import numpy as np

import math
import torch

from sklearn.cluster import KMeans
from collections import OrderedDict
from typing import Optional

from torch import nn, Tensor
from torch.nn import Linear, Conv1d, LayerNorm, DataParallel, ReLU, Sequential, Parameter
from torch.nn.functional import glu

from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros

from torch_geometric.utils import dense_to_sparse, scatter, add_remaining_self_loops, spmm
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor, torch_sparse
from torch_geometric.nn.dense.linear import Linear as pygLinear
import geoad.nn.recurrent_autoencoder as rae

from torch_sparse import eye as speye

##############################################################################################################    

class AEconv1D(nn.Module):

    # > Conv1d > MLP encoder > MLP decoder > ConvTranspose1d

    def __init__(self,
                 conv1d_n_feats, conv1d_kernel_size, conv1d_stride,
                 n_timestamps,
                 n_encoding_layers,
                 reduction
                 ):
        
        super(AEconv1D, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=conv1d_n_feats,
                                kernel_size=conv1d_kernel_size, stride=conv1d_stride)
        
        self.L_in = n_timestamps
        self.L_out = math.floor((self.L_in - conv1d_kernel_size)/conv1d_stride + 1)


        self.L_out_T = (self.L_out-1)*conv1d_stride + conv1d_kernel_size
        self.convT1d = nn.ConvTranspose1d(in_channels=conv1d_n_feats, out_channels=1,
                                           kernel_size=conv1d_kernel_size, stride=conv1d_stride, 
                                           output_padding=self.L_in-self.L_out_T)       

        mlp_in = conv1d_n_feats*self.L_out

        mlp_in_list = []
        mlp_in_list.append(mlp_in)

        # Encoder
        encoder_layers = OrderedDict()
        encoder_layers['linear_0'] = Linear(mlp_in, round( mlp_in * reduction ))
        encoder_layers['relu_0'] = ReLU()

        for n in range(1, n_encoding_layers):
            mlp_in = round( mlp_in * reduction )
            mlp_in_list.append(mlp_in)

            encoder_layers[f'linear_{n}'] = Linear(mlp_in, round( mlp_in * reduction ))
            encoder_layers[f'relu_{n}'] = ReLU()
        
        mlp_in_list.append( round(mlp_in * reduction ))
        self.encoder = Sequential(encoder_layers)

        # Decoder

        decoder_layers = OrderedDict()
        for n in range(0, n_encoding_layers):
            decoder_layers[f'linear_{n}'] = Linear( mlp_in_list[-n-1], mlp_in_list[-n-2] )
            decoder_layers[f'relu_{n}'] = ReLU()
        
        self.decoder = Sequential(decoder_layers)

    def forward(self, x):

        x = x.unsqueeze(1)
        f = self.conv1d(x)
        f = f.reshape((f.shape[0],-1))

        z = self.encoder(f)
        o = self.decoder(z)

        o = o.reshape((o.shape[0], -1, self.L_out ))
        r = self.convT1d(o)

        r = r.squeeze()

        return r


    def reset_parameters(self):
        self.conv1d.reset_parameters()
        self.convT1d.reset_parameters()
        for layer in self.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

##############################################################################################################    

class AE(nn.Module):

    # >  MLP encoder > MLP decoder 

    def __init__(self,
                 n_timestamps,
                 n_encoding_layers,
                 reduction
                 ):
        
        super(AE, self).__init__() 

        mlp_in = n_timestamps

        mlp_in_list = []
        mlp_in_list.append(mlp_in)

        # Encoder
        encoder_layers = OrderedDict()
        encoder_layers['linear_0'] = Linear(mlp_in, round( mlp_in * reduction ))
        encoder_layers['relu_0'] = ReLU()

        for n in range(1, n_encoding_layers):
            mlp_in = round( mlp_in * reduction )
            mlp_in_list.append(mlp_in)

            encoder_layers[f'linear_{n}'] = Linear(mlp_in, round( mlp_in * reduction ))
            encoder_layers[f'relu_{n}'] = ReLU()
        
        mlp_in_list.append( round(mlp_in * reduction ))
        self.encoder = Sequential(encoder_layers)

        # Decoder

        decoder_layers = OrderedDict()
        for n in range(0, n_encoding_layers-1):
            decoder_layers[f'linear_{n}'] = Linear( mlp_in_list[-n-1], mlp_in_list[-n-2] )
            decoder_layers[f'relu_{n}'] = ReLU()

        decoder_layers[f'linear_out'] = Linear( mlp_in_list[1], mlp_in_list[0] )
        
        self.decoder = Sequential(decoder_layers)

    def forward(self, x):

        z = self.encoder(x)
        o = self.decoder(z)

        return o


    def reset_parameters(self):

        for layer in self.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


##############################################################################################################

def get_layer_dims(dim_in, n_layers, reduction):
    layer_dims = [dim_in]
    for n in range(n_layers):
        layer_dims.append(round(layer_dims[-1]*reduction))
    return layer_dims
                
class GCNencoder(nn.Module):
    def __init__(self, dim_in, n_encoding_layers, reduction, conv_layer=GCNConv, conv_params=None):
        super(GCNencoder, self).__init__()

        if conv_params is None:
            conv_params = {}

        self.conv_layer = conv_layer
        self.layers = nn.ModuleList()
        for n in range(n_encoding_layers):
            self.layers.append(self.conv_layer(dim_in, round(dim_in * reduction), **conv_params))
            self.layers.append(nn.ReLU())
            dim_in = round(dim_in * reduction)

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            if isinstance(layer, self.conv_layer):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x                

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class GCNdecoder(nn.Module):
    def __init__(self, dim_in, n_layers, reduction, conv_layer=GCNConv, conv_params=None):
        super(GCNdecoder, self).__init__()

        if conv_params is None:
            conv_params = {}

        layer_dims = get_layer_dims(dim_in, n_layers, reduction)

        self.conv_layer = conv_layer
        self.layers = nn.ModuleList()
        for n in range(n_layers-1):
            self.layers.append(self.conv_layer(layer_dims[-n-1], layer_dims[-n-2], **conv_params ))
            self.layers.append(nn.ReLU())

        self.layers.append(self.conv_layer(layer_dims[1], layer_dims[0], **conv_params ))

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            if isinstance(layer, self.conv_layer):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x                

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()                

class GCN2MLP(nn.Module):

    # Asymmetric Graph Autoencoder with MLP decoder

    def __init__(self,
                 n_timestamps,
                 n_encoding_layers,
                 reduction
                 ):
        
        super(GCN2MLP, self).__init__()


        # Pre-computing dimensions of all layers
        self.layer_dims = get_layer_dims(n_timestamps, n_encoding_layers, reduction)

        # GCN encoder
        self.encoder = GCNencoder(n_timestamps, n_encoding_layers, reduction)

        # MLP decoder
        decoder_layers = OrderedDict()
        for n in range(0, n_encoding_layers-1):
            decoder_layers[f'linear_{n}'] = Linear( self.layer_dims[-n-1], self.layer_dims[-n-2] )
            decoder_layers[f'relu_{n}'] = ReLU()

        decoder_layers[f'linear_out'] = Linear( self.layer_dims[1], self.layer_dims[0] )
        self.decoder = Sequential(decoder_layers)

    def forward(self, x, edge_index, edge_weight=None):

        z = self.encoder(x, edge_index, edge_weight)
        o = self.decoder(z)

        return o

    def reset_parameters(self):

        self.encoder.reset_parameters()

        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class GCNAE(nn.Module):

    # Asymmetric Graph Autoencoder with GCN decoder

    def __init__(self,
                 n_timestamps,
                 n_encoding_layers,
                 reduction
                 ):
        
        super(GCNAE, self).__init__()

        # Pre-computing dimensions of all layers
        self.layer_dims = get_layer_dims(n_timestamps, n_encoding_layers, reduction)

        # GCN encoder
        self.encoder = GCNencoder(n_timestamps, n_encoding_layers, reduction)
        self.decoder = GCNdecoder(n_timestamps, n_encoding_layers, reduction)

    def forward(self, x, edge_index, edge_weight=None):

        z = self.encoder(x, edge_index, edge_weight)
        o = self.decoder(z, edge_index, edge_weight)

        return o

    def reset_parameters(self):

        self.encoder.reset_parameters()

        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


##############################################################################################################

# GALA IMPLEMENTATION

def gcn_norm(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 1.

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


def gcn_sharp(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2.

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)
    
    edge_weight_neg = -1*edge_weight

    edge_index_Ahat, edge_weight_Ahat = add_remaining_self_loops(edge_index, edge_weight_neg, fill_value, num_nodes)
    edge_index_Dhat, edge_weight_Dhat = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index_Dhat[0], edge_index_Dhat[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight_Dhat, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight_norm = deg_inv_sqrt[row] * edge_weight_Ahat * deg_inv_sqrt[col]

    return edge_index_Ahat, edge_weight_norm


class GCNGala(MessagePassing):
    r"""
    Adapted from pytorch geometric's GCNConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = 'smooth',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type

        self.lin = pygLinear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.norm_type == 'smooth':
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), self.flow, x.dtype)
        elif self.norm_type =='sharp':
            edge_index, edge_weight = gcn_sharp(edge_index, edge_weight, x.size(self.node_dim), self.flow, x.dtype)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
    

class GALA(nn.Module):

    # Symmetric Graph Autoencoder

    def __init__(self,
                 n_timestamps,
                 n_encoding_layers,
                 reduction
                 ):
        
        super(GALA, self).__init__()

        # Pre-computing dimensions of all layers
        self.layer_dims = get_layer_dims(n_timestamps, n_encoding_layers, reduction)

        # GCN encoder
        self.encoder = GCNencoder(n_timestamps, n_encoding_layers, reduction, GCNGala, {'norm_type':'smooth'})
        self.decoder = GCNdecoder(n_timestamps, n_encoding_layers, reduction, GCNGala, {'norm_type':'sharp'})

    def forward(self, x, edge_index, edge_weight=None):

        z = self.encoder(x, edge_index, edge_weight)
        o = self.decoder(z, edge_index, edge_weight)

        return o

    def reset_parameters(self):

        self.encoder.reset_parameters()

        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


##############################################################################################################

class RGEncoder(nn.Module):
    """Recurrent graph encoder"""

    def __init__(self, n_nodes, latent_dim, rnn, conv_params, device):
        super().__init__()

        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.device = device

        self.rec_enc1 = rnn(1, latent_dim, **conv_params)

    def forward(self, X, edge_index, edge_weight=None):

        seq_len = X.shape[1]
        edge_weight = torch.ones((edge_index.size(1), )) if edge_weight is None else edge_weight
    
        H_i = torch.zeros((self.n_nodes, self.latent_dim), device=self.device)

        for i in range(0, seq_len):
            out = self.rec_enc1(X[:,i].unsqueeze(1), edge_index, edge_weight, H_i)
            H_i = out[0] if isinstance(out, tuple) else out

        return H_i

class RGDecoder(nn.Module):
    """Recurrent graph decoder for RNN and GRU"""

    def __init__(self, latent_dim, rnn, conv_params, device):
        super().__init__()

        self.device = device
        self.rec_dec1 = rnn(1, latent_dim, **conv_params)
        self.dense_dec1 = nn.Linear(latent_dim, 1)

    def forward(self, H_0, edge_index, edge_weight, seq_len):
        # Initialize output
        X = torch.tensor([], device = self.device)

        H_i = H_0
        x_i = self.dense_dec1(H_0)
        # X = torch.cat([X, x_i], axis=1) # If using this, use range(1, seq_len). Else, use range(0, seq_len)

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            out = self.rec_dec1(x_i, edge_index, edge_weight, H_i)
            H_i = out[0] if isinstance(out, tuple) else out
            x_i = self.dense_dec1(H_i)
            X = torch.cat([X, x_i], axis=1)

        return X

class RecurrentGAE(nn.Module):
    """Recurrent graph autoencoder. For single feature """

    def __init__(self, n_nodes, latent_dim, rnn, conv_params, device):
        super().__init__()

        if conv_params is None:
            conv_params = {}

        # Encoder and decoder configuration
        self.device = device

        # Encoder and decoder
        self.encoder = RGEncoder(n_nodes, latent_dim, rnn, conv_params, device)
        self.decoder = RGDecoder(latent_dim, rnn, conv_params, device)

    def forward(self, X, edge_index, edge_weight):
        seq_len = X.shape[1]
        H_enc = self.encoder(X, edge_index, edge_weight)
        out = self.decoder(H_enc, edge_index, edge_weight, seq_len)

        return torch.flip(out, [1]).squeeze()
    
    def reset_parameters(self):

        for layer in self.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()




##############################################################################################################
def columnvec(X):
    return X.t().contiguous().view(-1).unsqueeze(1)

def inversevec(x, shape):
    return x.view(shape[::-1]).t()

def get_layer_timestamps(T, n_layers, r):
    timestamps = [T]
    for i in range(n_layers):
        timestamps.append(int(np.ceil(timestamps[-1]/r)))
    return timestamps

def spkron(A1, A2):
    edge_index1, edge_value1 = A1.indices(), A1.values()
    edge_index2, edge_value2 = A2.indices(), A2.values()
    
    row1, col1 = edge_index1
    row2, col2 = edge_index2

    # Compute the Kronecker product of the indices
    kronecker_row = row1.view(-1, 1) * A2.size(0) + row2.view(1, -1)
    kronecker_col = col1.view(-1, 1) * A2.size(1) + col2.view(1, -1)

    # Compute the Kronecker product of the values
    kronecker_value = edge_value1.view(-1, 1) * edge_value2.view(1, -1)

    # Compute the size of the resulting tensor
    size = (A1.size(0)*A2.size(0), A1.size(1)*A2.size(1))

    # Flatten and return
    return torch.sparse_coo_tensor(indices=torch.stack([kronecker_row.flatten(), kronecker_col.flatten()]),
                                   values=kronecker_value.flatten(),
                                   size=size)

def sppow(A,k):
    if k==0:
        A0 = speye(A.size()[0])
        return torch.sparse_coo_tensor(indices=A0[0], values=A0[1], size=(A.size())).coalesce()
    elif k==1:
        return A
    elif k>1:
        return A.matrix_power(k)
    
# ---------                              ---------------#
class GTConvFilter(nn.Module):
    def __init__(self, K):
        super(GTConvFilter, self).__init__()

        self.K = K
        self.h = torch.nn.Parameter(torch.randn(self.K))

    def forward(self, x, S_powers):

        # SPARSE
        H = torch.zeros(S_powers[0].size()).to_sparse().to(S_powers[0].device)
        for i in range(self.K):
            H += self.h[i]*S_powers[i]
        return H.matmul(x)
    
    def reset_parameters(self):
        self.h = torch.nn.Parameter(torch.randn(self.K))

class GTConvFilterDense(nn.Module):
    def __init__(self, K):
        super(GTConvFilterDense, self).__init__()

        self.K = K
        self.h = torch.nn.Parameter(torch.randn(self.K))

    def forward(self, x, S_powers):

        # DENSE
        H = sum(self.h[i]*S_powers[i] for i in range(self.K))
        return H.matmul(x)
    
    def reset_parameters(self):
        self.h = torch.nn.Parameter(torch.randn(self.K))

# ---------                              ---------------#

class GTConvBank(nn.Module):
    def __init__(self, in_channels, K):
        super(GTConvBank, self).__init__()
        
        self.K = K
        self.in_channels = in_channels
        
        self.filters = nn.ModuleList()
        for i in range(in_channels):
            self.filters.append(GTConvFilter(self.K))

    def forward(self, X, S_powers):
        if X.shape[-1]!=self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels. Found {X.shape[-1]}")

        y = sum(filter(X[:,i], S_powers) for (i,filter) in enumerate(self.filters))
        return y
    
    def reset_parameters(self):
        for filter in self.filters:
            filter.reset_parameters()

class GTConvBankDense(nn.Module):
    def __init__(self, in_channels, K):
        super(GTConvBankDense, self).__init__()
        
        self.K = K
        self.in_channels = in_channels
        
        self.filters = nn.ModuleList()
        for i in range(in_channels):
            self.filters.append(GTConvFilterDense(self.K))

    def forward(self, X, S_powers):
        if X.shape[-1]!=self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels. Found {X.shape[-1]}")

        y = sum(filter(X[:,i], S_powers) for (i,filter) in enumerate(self.filters))
        return y
    
    def reset_parameters(self):
        for filter in self.filters:
            filter.reset_parameters()

# ---------                              ---------------#

class GTConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(GTConvLayer, self).__init__()

        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.banks = nn.ModuleList()

        for i in range(out_channels):
            self.banks.append(GTConvBank(self.in_channels, self.K))
        
    def forward(self, X, S_powers):
        '''
        X (NT, in_channels)
        '''
        if X.shape[-1]!=self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels. Found {X.shape[-1]}")

        Y = torch.stack([bank(X, S_powers) for bank in self.banks], dim=-1)
        return Y
    

class GTConvLayerDense(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(GTConvLayerDense, self).__init__()

        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.banks = nn.ModuleList()

        for i in range(out_channels):
            self.banks.append(GTConvBankDense(self.in_channels, self.K))
        
    def forward(self, X, S_powers):
        '''
        X (NT, in_channels)
        '''
        if X.shape[-1]!=self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels. Found {X.shape[-1]}")

        Y = torch.stack([bank(X, S_powers) for bank in self.banks], dim=-1)
        return Y
    
# ---------                              ---------------#

class Downsampler(nn.Module):
    def __init__(self, r, agg):
        super(Downsampler, self).__init__()

        self.r = r
        if agg not in [torch.sum, torch.mean, torch.min, torch.max, 'first', 'last']:
            raise Exception('Invalid aggregation function')
        
        if agg == 'first':
            def agg(tensor, dim):
                return tensor.select(dim, 0)
        elif agg == 'last':
            def agg(tensor, dim):
                return tensor.select(dim, -1)
        
        self.agg = agg

    def downsampling(self, matrix):
        # returns downsampled matrix
        N, T = matrix.shape


        remainder = T % self.r
        if remainder != 0:
            # If dimensions do not match, pad the matrix by repeating the last column
            padding = torch.repeat_interleave(matrix[:, -1:], self.r - remainder, dim=1)
            matrix = torch.cat((matrix, padding), dim=1)

        reshaped = matrix.view(N, -1, self.r)
        downsampled = self.agg(reshaped, dim=2)
        return downsampled.values
        
    def forward(self, X, shape):
        # input is (NT x F)
        # output is (N*(T/r) x F)
        # shape is (N,T)

        n_features = X.shape[-1]
        Xd = torch.stack([columnvec(  self.downsampling(inversevec(X[:,i], shape)) ).squeeze()
                          for i in range(n_features)], 
                          dim=-1)

        return Xd
    
class Upsampler(nn.Module):
    def __init__(self, r):
        super(Upsampler, self).__init__()

        self.r = r

    def upsampling(self, matrix, target_T):
        N = matrix.shape[0]
        upsampled = torch.zeros((N, target_T)).to(matrix.device)
        upsampled[:, ::self.r] = matrix
        return upsampled
    
    def forward(self, Y, shape, target_T):
        n_features = Y.shape[-1]
        Z = torch.stack([columnvec(  self.upsampling( inversevec(Y[:,i], shape), target_T )  ).squeeze()
                         for i in range(n_features)],
                         dim=-1)
        return Z

# ---------                              ---------------#
   
class GTConvAE(nn.Module):
    def __init__(self, hidden_features, K, r, temp_graph, device, dense=True):
        super().__init__()

        self.K = K
        self.r = r
        self.temp_graph = temp_graph
        self.device = device
        self.s = self.parameter_s()
        self.dense = dense

        self.n_layers = len(hidden_features)
        hidden_features.insert(0,1)

        if self.dense:
            conv = GTConvLayerDense
        else:
            conv = GTConvLayer

        self.encoder_layers = OrderedDict()
        self.decoder_layers = OrderedDict()
        for n in range(1, self.n_layers+1):
            self.encoder_layers[f'conv_{n}'] = conv(hidden_features[n-1], hidden_features[n], K)
            self.encoder_layers[f'downsampling_{n}'] = Downsampler(r=r, agg=torch.max)
            self.encoder_layers[f'activation_{n}'] = ReLU()

            self.decoder_layers[f'upsampling_{n}'] = Upsampler(r=r)
            self.decoder_layers[f'activation_{n}'] = ReLU()
            self.decoder_layers[f'conv_{n}'] = conv(hidden_features[-n], hidden_features[-n-1], K)


    def forward(self, X, Sg):
        N = X.shape[0]
        T = X.shape[1]

        x = columnvec(X)

        Ts = get_layer_timestamps(T, self.n_layers, self.r)
        S_powers_per_layer = []
        for t in Ts:
            St = self.create_temp_graph(t, self.temp_graph).to(self.device)
            if self.dense:
                S_powers_per_layer.append(self.ProductGraphPowersDense(St, Sg, self.s))
            else:
                S_powers_per_layer.append(self.ProductGraphPowers(St, Sg, self.s))

        for n in range(1, self.n_layers+1):
            S_powers = S_powers_per_layer[n-1]
            x = self.encoder_layers[f'conv_{n}'](x, S_powers)
            x = self.encoder_layers[f'downsampling_{n}'](x, (N,Ts[n-1]))
            x = self.encoder_layers[f'activation_{n}'](x)
        
        for n in range(1, self.n_layers+1):
            S_powers = S_powers_per_layer[-n-1]
            x = self.decoder_layers[f'upsampling_{n}'](x, (N,Ts[-n]), Ts[-n-1] )
            x = self.decoder_layers[f'activation_{n}'](x)
            x = self.decoder_layers[f'conv_{n}'](x, S_powers)

        return inversevec(x, (N,T)).to_dense().contiguous()

    def ProductGraphPowers(self, St, Sg, s):

        S = torch.zeros(size=(St.shape[0]*Sg.shape[0], St.shape[1]*Sg.shape[1])).to_sparse().to(self.device)
        for i in range(2):
            for j in range(2):
                S = S + s[i,j]*spkron(St.matrix_power(i).to_sparse(), Sg.matrix_power(j).to_sparse())

        S_powers = []
        for k in range(self.K+1):
            S_powers.append(sppow(S.to_sparse(),k).to(self.device))
        return S_powers
    
    def ProductGraphPowersDense(self, St, Sg, s):

        S = sum(s[i,j]*torch.kron(St.matrix_power(i), Sg.matrix_power(j)) for i in range(2) for j in range(2))
        S_powers = []
        for k in range(self.K+1):
            S_powers.append(S.matrix_power(k))
        return S_powers

    @staticmethod
    def create_temp_graph(t, temp_graph):
        if temp_graph =='cyclic_directed':
            return torch.eye(t).roll(1, dims=-1).t()
        
        elif temp_graph =='line_directed':
            return torch.diag(torch.ones((t-1)), diagonal=1)
        
        elif temp_graph =='cyclic_undirected':
            return torch.eye(t).roll(-1, dims=-1).t() + torch.eye(t).roll(1, dims=-1).t()
        
        elif temp_graph =='line_undirected':
            return torch.diag(torch.ones((t-1)), diagonal=1) + torch.diag(torch.ones((t-1)), diagonal=-1)

        else:
            print("Invalid type. Must be one of (cyclic or line) + _ + (directed or undirected)")

    def reset_parameters(self):
        for n in range(1, self.n_layers+1):
            for bank in self.encoder_layers[f'conv_{n}'].banks:
                for filter in bank.filters:
                    filter.reset_parameters()
            for bank in self.decoder_layers[f'conv_{n}'].banks:
                for filter in bank.filters:
                    filter.reset_parameters()

        self.s = self.parameter_s()
    
    @staticmethod
    def parameter_s():
        return torch.nn.Parameter(torch.tensor( ((0.05,0.05), (0.05,0.05))) + 0.05*torch.randn((2,2)) )
    


########## ALIASES ##########

RAE = rae.RecurrentAE
GUNet = GraphUNet

##############################################################################################################
def main():
    return 0


if __name__ == "__main__":
    main()
