import typing as t

from torch import nn

from layers import *


class GraphInf(nn.Module):
    def __init__(
        self,
        num_in_feat: int,
        num_k_feat: int,
        num_botnec_feat: int,
        causal_hidden_sizes: t.Iterable,

    ):
        self.num_in_feat = num_in_feat
        self.dense1 = DenseNet(
            num_feat=self.num_in_feat,
            num_k_feat=self.num_k_feat,
            num_botnec_feat=self.num_botnec_feat,
            causal_hidden_sizes=self.causal_hidden_sizes,
            num_dense_layers=self.num_dense_layers,
            activation=self.activation
        )

        self.dense2 = DenseNet(
            num_feat=self.num_in_feat,
            num_k_feat=self.num_k_feat,
            num_botnec_feat=self.num_botnec_feat,
            causal_hidden_sizes=self.causal_hidden_sizes,
            num_dense_layers=self.num_dense_layers,
            activation=self.activation
        )
        
        self.fc1 = nn.Linear(
            self.num_k_feat,
            self.num_k_feat
        )

        self.fc2 = nn.Linear(
            self.num_k_feat,
            self.num_k_feat
        )

        # self.weave1 = WeaveLayer(
        #     self.num_k_feat,
        #     self.num_k_feat
        # )
        
        # self.weave2 = WeaveLayer(
        #     self.num_k_feat,
        #     self.num_k_feat
        # )

    def encode(self, feat, adj):
        h1 = self.dense1(x)
        return self.fc1(h1), self.fc2(h2)

    def reparametrize(
        self,
        mu,
        logvar
    ): 
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(
        self,
        feat_o: torch.Tensor,
        feat_c: torch.Tensor,
        adj_o: torch.sparse.Tensor,
        adj_c: torch.sparse.Tensor
    ):
        z = self.encode(feat_o, adj_o)

