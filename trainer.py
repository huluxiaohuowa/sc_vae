import time

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
# device_ids = [1,2,3]

num_epochs = 10
t = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))

try:
    with SummaryWriter(f'./events/{t}/') as writer:
        for epoch in ipb(range(num_epochs), desc="epochs"):
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
            # model = nn.DataParallel(model, device_ids=device_ids)
            model = model.to(device)
            model.train()

            optim = adabound.AdaBound(
                model.parameters(),
                lr=1e-3,
                final_lr=0.1
            )

            dataloader = Dataloader(batch_size=batch_size)

            for step, (s, c) in enumerate(
                ipb(
                    dataloader,
                    desc="step",
                    total=dataloader.num_id_block
                )
            ):
                s_nfeat, s_einfo, s_adj = graph_to_whole_graph(
                    s.adjacency_matrix(),
                    torch.stack(s.edges(), dim=0),
                    s.ndata['feat'],
                    s.edata['feat']
                )
                c_nfeat, c_einfo, c_adj = graph_to_whole_graph(
                    c.adjacency_matrix(),
                    torch.stack(c.edges(), dim=0),
                    c.ndata['feat'],
                    c.edata['feat']
                )
                s_nfeat, s_adj, c_nfeat = (
                    s_nfeat.to(device), s_adj.to(device), c_nfeat.to(device)
                )
                x_recon, mu1, logvar1, mu2, logvar2 = (
                    model(s_nfeat, c_nfeat, s_adj)
                )
                optim.zero_grad()
                feat = label_to_onehot(s_nfeat, 39)
                MSE, KL = loss_func(
                    x_recon, feat, mu1, logvar1, mu2, logvar2
                )
                loss = MSE + KL
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
                # print(loss, MSE, KL)
            torch.save(model, f'./ckpt/{t}/cktp_{str(epoch)}')

except KeyboardInterrupt:
    torch.save(model, f'./ckpt/{t}/cktp_{str(epoch)}')
