# import typing as t

import torch
from torch import nn
import torch_scatter
from torch.utils import checkpoint as tuc

import ops


__all__ = [
    "WeaveLayer",
    "DenseNet",
    "AvgPooling",
    "SumPooling"
]


def _bn_function_factory(bn_module):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, -1)
        bottleneck_output = bn_module(concated_features)
        return bottleneck_output

    return bn_function


class BNReLULinear(nn.Module):
    """
    Linear layer with bn->relu->linear architecture
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = 'elu'
    ):
        """

        Args:
            in_features (int):
                The number of input features
            out_features (int):
                The number of output features
            activation (str):
                The type of activation unit to use in this module,
                default to elu
        """
        super(BNReLULinear, self).__init__()
        self.bn_relu_linear = nn.Sequential(
            nn.BatchNorm1d(in_features),
            ops.get_activation(
                activation,
                inplace=True
            ),
            nn.Linear(
                in_features,
                out_features,
                bias=False)
            )

    def forward(self, x):
        """The forward method"""
        return self.bn_relu_linear(x)


class WeaveLayer(nn.Module):
    def __init(
        self,
        # num_v_in_feat: int,
        # num_v_out_feat: int,
        # num_e_in_feat: int,
        # num_e_out_feat: int,
        num_in_feat: int,
        num_out_feat: int,
        activation: str='relu',
    ):
        super().__init__()
        # self.num_v_in_feat = num_v_in_feat
        # self.num_v_out_feat = num_v_out_feat
        # self.num_e_in_feat = num_e_in_feat
        # self.num_e_out_feat = num_e_out_feat
        self.num_in_feat = num_in_feat
        self.num_out_feat = num_out_feat
        self.activation = activation
        if self.activation is not None:
            self.linear = BNReLULinear(
                self.in_features,
                self.out_features,
                self.activation
            )
            # self.elinear = BNReLULinear(
            #     self.in_features,
            #     self.out_features,
            #     self.activation
            # )

        else:
            self.linear = nn.Linear(
                self.in_features,
                self.out_features * self.num_bond_types,
                self.activation
            )
            # self.elinear = nn.Linear(
            #     self.in_features,
            #     self.out_features * self.num_bond_types,
            #     self.activation
            # )

    def forward(
        self,
        n_feat: torch.Tensor,
        adj: torch.Tensor
    ):
        assert adj.is_sparse
        n_feat = self.linear(n_feat)
        n_feat_self = n_feat * n_feat
        n_feat_adj = torch.mm(adj, n_feat)

        return n_feat_self + n_feat_adj


class CasualWeave(nn.Module):
    def __init__(
        self,
        num_feat: int,
        hidden_sizes: t.Iterable,
        activation: str='elu'
    ):
        self.num_feat = num_feat
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation

        layers = []
        for i, (in_feat, out_feat) in enumerate(
            zip(
                [self.num_feat, ] +
                list(self.hidden_sizes)[:-1],  # in_features
                self.hidden_sizes  # out_features
            )
        ):
            if i == 0:
                layers.append(
                    WeaveLayer(
                        in_feat,
                        out_feat,
                        None
                    )
                )
            else:
                layers.append(
                    WeaveLayer(
                        in_feat,
                        out_feat,
                        self.activation
                    )
                )
            self.layers = nn.ModuleList(layers)

    def forward(
        self,
        feat: torch.Tensor,
        adj: torch.Tensor
    ):
        feat_out = feat
        for layer in self.layers:
            feat_out = layer(
                feat_out,
                adj
            )
        return feat_out


class DenseLayer(nn.Module):
    def __init__(
        self,
        num_in_feat: int,
        num_botnec_feat: int,
        num_out_feat: int,
        activation: str = 'elu',
    ):
        self.num_in_feat = num_in_feat
        self.num_out_feat = num_out_feat
        self.num_botnec_feat = num_botnec_feat
        self.activation = activation

        self.bottlenec = BNReLULinear(
            self.num_in_feat,
            self.num_botnec_feat,
            self.activation
        )

        self.weave = WeaveLayer(
            self.num_botnec_feat,
            self.num_out_feat,
            self.activation
        )

    def forward(
        self,
        ls_feat: t.List[torch.Tensor],
        adj: torch.Tensor,
    ):
        bn_fn = _bn_function_factory(self.bottlenec)
        feat = tuc.checkpoint(bn_fn, *ls_feat)
        return self.weave(
            feat,
            adj
        )
        
        
class DenseNet(nn.Module):
    def __init__(
        self,
        num_feat: int,
        causal_hidden_sizes: t.Iterable,
        num_botnec_feat: int,
        num_k_feat: int,
        num_dense_layers: int,
        num_out_feat: int,
        activation: str='elu'
    ):
        self.num_feat = num_feat
        self.num_dense_layers = num_dense_layers
        self.causal_hidden_sizes = list(causal_hidden_sizes)
        self.num_out_feat = num_out_feat
        self.casual = CasualWeave(
            self.num_feat,
            self.causal_hidden_sizes,
            self.activation
        )
        dense_layers = []
        for i in range(self.num_dense_layers):
            dense_layers.append(
                DenseLayer(
                    self.causal_hidden_sizes[-1] + i * self.num_k_feat,
                    self.num_botnec_feat,
                    self.num_k_feat,
                    self.activation
                )
            )
        self.dense_layers = nn.ModuleList(dense_layers)

        self.output = BNReLULinear(
            (
                self.causal_hidden_sizes[-1] +
                self.num_dense_layers * self.num_k_feat
            ),
            self.num_out_feat,
            self.activation
        )
        
    def forward(
        self,
        feat,
        adj
    ):
        feat = self.casual(
            feat,
            adj
        )
        ls_feat = [feat, ]
        for dense_layer in self.dense_layers:
            feat_i = dense_layer(
                ls_feat,
                adj
            )
            ls_feat.append(feat_i)
        feat_cat = torch.cat(ls_feat, dim=-1)
        return self.output(feat_cat)

# class GraphConv(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         num_bond_types: int,
#         activation: str='elu',
#     ):
#         """Summary
        
#         Args:
#             in_features (int): Description
#             out_features (int): Description
#             num_bond_types (int): Description
#             activation (str, optional): Description
#         """
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.activation = activation
#         self.num_bond_types = num_bond_types
#         if self.activation is not None:
#             self.linear = BNReLULinear(
#                 self.in_features,
#                 self.out_features,
#                 self.activation
#             )
#         else:
#             self.linear = nn.Linear(
#                 self.in_features,
#                 self.out_features * self.num_bond_types,
#                 self.activation
#             )

#     def forward(
#         self,
#         atom_features: torch.Tensor,
#         adj: torch.Tensor,  # size num_atoms x num_atoms*num_bond_types
#     ):
#         # size: batch_size x num_bond_types*out_features
#         atom_features = self.linear(atom_features)
#         # size: batch_size x num_bond_types x out_features
#         atom_features = atom_features.view(
#             -1, 
#             self.num_bond_types,
#             self.out_features
#         ).transpose(
#             0, 1  # size: num_bond_types x num_atoms x out_features
#         ).contiguous(
#         ).view(
#             -1,
#             self.out_features
#         )  # size num_bond_types*batch_size x out_features
#         out = torch.sparse.mm(adj, atom_features)
#         return out  # size: batch_size x out_features
        
#     def __repr__(self):
#         return (
#             self.__class__.__name__ + ' (' +
#             str(self.in_features) + ' -> ' +
#             str(self.out_features) + ')'
#         )


class _Pooling(nn.Module):
    def __init__(
        self,
        in_features: int,
        pooling_op: t.Callable=torch_scatter.scatter_mean,
        activation: str='elu'
    ):
        """Summary
        Args:
            in_features (int): Description
            pooling_op (t.Callable, optional): Description
            activation (str, optional): Description
        """
        super(_Pooling, self).__init__()
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(in_features),
            ops.get_activation(activation, inplace=True)
        )
        self.pooling_op = pooling_op

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        num_seg: int=None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor, size=[N, in_features]
            ids (torch.Tensor): A tensor of type `torch.long`, size=[N, ]
            num_seg (int): The number of segments (graphs)
        Returns:
            torch.Tensor: Output tensor with size=[num_seg, in_features]
        """

        # performing batch_normalization and activation
        x_bn = self.bn_relu(x)  # size=[N, in_features]

        # performing segment operation
        x_pooled = self.pooling_op(
            x_bn,
            dim=0,
            index=ids,
            dim_size=num_seg
        )  # size=[num_seg, in_features]

        return x_pooled


class AvgPooling(_Pooling):
    """Average pooling layer for graph"""

    def __init__(
        self,
        in_features: int,
        activation: str = 'elu'
    ):
        """ Performing graph level average pooling (with bn_relu)

        Args:
            in_features (int):
                The number of input features
            activation (str):
                The type of activation function to use, default to elu
        """
        super(AvgPooling, self).__init__(
            in_features,
            activation=activation,
            pooling_op=torch_scatter.scatter_mean
        )


class SumPooling(_Pooling):
    """Sum pooling layer for graph"""

    def __init__(
        self,
        in_features: int,
        activation: str = 'elu'
    ):
        """ Performing graph level sum pooling (with bn_relu)

        Args:
            in_features (int):
                The number of input features
            activation (str):
                The type of activation function to use, default to elu
        """
        super(SumPooling, self).__init__(
            in_features,
            activation=activation,
            pooling_op=torch_scatter.scatter_add
        )

