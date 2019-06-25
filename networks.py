import typing as t

from torch import nn

from layers import *


class GraphInf(nn.Module):
    def __init__(
        self,
        num_in_feat: int,  # 
        num_c_in_feat: int,
        num_embeddings: int,  # 
        causal_hidden_sizes: t.Iterable,  # 
        num_botnec_feat: int,  # 16 x 4
        num_k_feat: int,  # 16
        num_dense_layers: int,
        num_out_feat: int,
        num_z_feat: int,
        activation: str='elu',
        use_cuda: bool=True
    ):
        
        self.num_in_feat = num_in_feat
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_in_feat, num_embeddings)
        self.c_embedding = nn.Embedding(num_c_feat, num_embeddings)
        self.causal_hidden_sizes = causal_hidden_sizes
        self.num_botnec_feat = num_botnec_feat
        self.num_k_feat = num_k_feat
        self.num_dense_layers = num_dense_layers
        self.num_out_feat = num_out_feat

        self.dense1 = DenseNet(  # for encoding
            num_feat=self.embedding,
            causal_hidden_sizes=self.causal_hidden_sizes,
            num_botnec_feat=self.num_botnec_feat,
            num_k_feat=self.num_k_feat,
            num_dense_layers=self.num_dense_layers,
            num_out_feat=self.num_out_feat,
            activation=self.activation 
        )

        self.dense2 = DenseNet(  # for decoding
            num_feat=self.num_z_feat,
            causal_hidden_sizes=self.causal_hidden_sizes,
            num_botnec_feat=self.num_botnec_feat,
            num_k_feat=self.num_k_feat,
            num_dense_layers=self.num_dense_layers,
            num_out_feat=self.num_in_feat,
            activation=self.activation 
        )

        self.c_dense = DenseNet(
            num_feat=self.num_embeddings,
            causal_hidden_sizes=self.causal_hidden_sizes,
            num_botnec_feat=self.num_botnec_feat,
            num_k_feat=self.num_k_feat,
            num_dense_layers=self.num_dense_layers,
            num_out_feat=self.num_out_feat
        )
        
        self.fc1 = nn.Linear(
            self.num_out_feat,
            self.num_z_feat
        )

        self.fc2 = nn.Linear(
            self.num_out_feat,
            self.num_z_feat
        )

        self.c_fc = nn.Linear(
            self.num_out_feat,
            self.num_z_feat
        )

    def encode(self, feat, adj):
        h1 = self.dense1(feat, adj)
        return self.fc1(h1), self.fc2(h1)

    def decode(self, z):
        x_recon = self.dense2(z)
        return x_recon

    def reparametrize(
        self,
        mu,
        logvar
    ): 
        std = logvar.mul(0.5).exp_()
        if use_cuda and torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(
        self,
        feat_o: torch.Tensor,
        feat_c: torch.Tensor,
        adj: torch.sparse.Tensor,
    ):
        mu, logvar = self.encode(feat_o, adj)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z, adj)
        c_z = self.c_dense(feat_c, adj)
        





