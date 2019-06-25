import time

from ipypb import ipb
import torch
from torch.utils.tensorboard import SummaryWriter
import adabound

from data import *
from utils import *
from ops import *
from networks import *

batch_size = 256
device = torch.device('cuda:0')
use_cuda = True
nem_embeddings = 4
casual_hidden_sizes = [32, 16]
num_botnec_feat = 64
num_k_feat = 16
num_dense_layers = 4
num_out_feat = 4
num_z_feat = 4
activation = 'elu'
num_embeddings = 16

device_ids = [0, 1, 2]

num_epochs = 10
t = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))

try:
    with SummaryWriter(f'./events/{t}/') as writer:
        for epoch in ipb(range(num_epochs), decs="epochs"):
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
            model = nn.DataParallel(model, device_ids=device_ids)
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
                x_recon, mu1, logvar1, mu2, logvar2 = (
                    model(s_nfeat, c_nfeat, s_adj)
                )
                optim.zero_grad()
                loss = loss_func(recon_x, x, mu1, logvar1, mu2, logvar2)
                loss.backward()
                optim.step()
                writer.add_scalar(
                    f'loss_{str(epoch)}',
                    loss.item(),
                    step
                )
                torch.save(model, f'./ckpt/{t}/cktp_{str(epoch)}')

except KeyboardInterrupt:
    torch.save(model, f'./ckpt/{t}/cktp_{str(epoch)}')
