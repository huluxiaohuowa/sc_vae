import time
from os import path, makedirs

from ipypb import ipb
import torch
# from torch import nn
from torch.utils.tensorboard import SummaryWriter
import adabound

from data import *
from utils import *
from ops import *
from networks import *

batch_size = 128
device = torch.device('cuda:1')
use_cuda = True
num_embeddings = 8
casual_hidden_sizes = [16, 32]
num_botnec_feat = 72
num_k_feat = 24
num_dense_layers = 20
num_out_feat = 256
num_z_feat = 2
activation = 'elu'
LR = 1e-3
final_lr = 0.1
beta = 1

num_epochs = 10
t = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
model = GraphInf(
    num_in_feat=39,
    num_c_feat=4,
    num_embeddings=num_embeddings,
    casual_hidden_sizes=casual_hidden_sizes,
    num_botnec_feat=num_botnec_feat,  # 16 x 4
    num_k_feat=num_k_feat,  # 16
    num_dense_layers=num_dense_layers,
    num_out_feat=num_out_feat,
    num_z_feat=num_z_feat,
    activation=activation,
    use_cuda=use_cuda
)
optim = adabound.AdaBound(
    model.parameters(),
    lr=LR,
    final_lr=final_lr
)
# model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)
model.train()
try:
    with SummaryWriter(f'./events/{t}/') as writer:
        for epoch in ipb(range(num_epochs), desc="epochs"):
            if not path.exists(f'./ckpt/{t}/'):
                makedirs(f'./ckpt/{t}/')
            # optim = torch.optim.SGD(
            #     model.parameters(),
            #     lr=lr
            # )

            dataloader = Dataloader(batch_size=batch_size)

            for step, (s, c) in enumerate(
                ipb(
                    dataloader,
                    desc="step",
                    total=dataloader.num_id_block
                )
            ):
                # s_nfeat, s_einfo, s_adj = graph_to_whole_graph(
                #     s.adjacency_matrix(),
                #     torch.stack(s.edges(), dim=0),
                #     s.ndata['feat'],
                #     s.edata['feat']
                # )
                # c_nfeat, c_einfo, c_adj = graph_to_whole_graph(
                #     c.adjacency_matrix(),
                #     torch.stack(c.edges(), dim=0),
                #     c.ndata['feat'],
                #     c.edata['feat']
                # )

                num_N = s.number_of_nodes()
                num_E = s.number_of_edges()
                adj = s.adjacency_matrix().coalesce()

                indices = torch.cat(
                    [adj.indices(), torch.arange(0, num_N).repeat(2, 1)],
                    dim=-1
                )

                values = torch.ones(num_E + num_N)

                s_adj = torch.sparse_coo_tensor(
                    indices,
                    values,
                    torch.Size([num_N, num_N])
                )

                s_nfeat = s.ndata['feat']
                c_nfeat = c.ndata['feat']
                s_nfeat, s_adj, c_nfeat = (
                    s_nfeat.to(device), s_adj.to(device), c_nfeat.to(device)
                )
                s_nfeat = onehot_to_label(s_nfeat)
                c_nfeat = onehot_to_label(c_nfeat)

                seg_id_block = [
                    torch.LongTensor([i]).repeat(j)
                    for i, j in enumerate(s.batch_num_nodes)
                ]

                seg_ids = torch.cat(seg_id_block, dim=-1)

                x_recon, mu1, logvar1, mu2, logvar2 = (
                    model(s_nfeat, c_nfeat, s_adj)
                )

                optim.zero_grad()
                MSE, KL = loss_func(
                    x_recon, s_nfeat, mu1, logvar1, mu2, logvar2, seg_ids
                )
                loss = MSE + beta * KL
                loss.backward()
                optim.step()
                writer.add_scalar(
                    f'loss_{str(epoch)}',
                    loss.cpu().item(),
                    step
                )
                writer.add_scalar(
                    f'MSE_{str(epoch)}',
                    MSE.cpu().item(),
                    step
                )
                writer.add_scalar(
                    f'KL_{str(epoch)}',
                    KL.cpu().item(),
                    step
                )
                # print(loss.item(), MSE.item(), KL.item())
            torch.save(model, f'./ckpt/{t}/cktp_{str(epoch)}')

except KeyboardInterrupt:
    torch.save(model, f'./ckpt/{t}/cktp_{str(epoch)}')
