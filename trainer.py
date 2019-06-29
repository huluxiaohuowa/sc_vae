# import time
import sys
import json
from os import path, makedirs
import typing as t
from multiprocessing import cpu_count

from ipypb import ipb
import torch
# from torch import nn
from torch.utils.tensorboard import SummaryWriter
import adabound

from data import *
from utils import *
from ops import *
from networks import *

# device_id = 1
# device = torch.device(f'cuda:{device_id}')
# batch_size = 128

# use_cuda = True
# num_embeddings = 8
# casual_hidden_sizes = [16, 32]
# num_botnec_feat = 72
# num_k_feat = 24
# num_dense_layers = 20
# num_out_feat = 256
# num_z_feat = 2
# activation = 'elu'
# LR = 1e-3
# final_lr = 0.1
# beta = 1.

# num_epochs = 10


def engine(
    ckpt_loc: str='ckpt-default',
    device_id: int=1,
    batch_size: int=128,
    use_cuda: bool=True,
    num_embeddings: int=8,
    casual_hidden_sizes: t.Iterable=[16, 32],
    num_botnec_feat: int=72,
    num_k_feat: int=24,
    num_dense_layers: int=20,
    num_out_feat: int=268,
    num_z_feat: int=2,
    activation: str='elu',
    LR: float=1e-3,
    final_lr: float=0.1,
    init_beta: float=0.,
    final_beta: float=1.,
    num_annealing_steps: int=2000,
    # beta: float=0.25,
    grad_clip=3.0,
    num_epochs: int=5,
    np=1
):
    beta_step_len = (final_beta - init_beta) / num_annealing_steps
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
    device = torch.device(f'cuda:{device_id}')
    model = model.to(device)
    model.train()
    save_loc = path.join(
        path.dirname(__file__),
        'ckpt',
        ckpt_loc
    )
    events_loc = path.join(save_loc, 'events')
    if not path.exists(events_loc):
        makedirs(events_loc)
    try:
        with SummaryWriter(events_loc) as writer:
            step = 0
            has_nan_or_inf = False
            for epoch in ipb(range(num_epochs), desc='epochs'):
                if has_nan_or_inf:
                    break
                dataloader = Dataloader(
                    num_workers=np,
                    batch_size=batch_size
                )
                for s, c, block in ipb(
                    dataloader,
                    desc="step",
                    total=dataloader.num_id_block
                ):
                    beta = min(init_beta + beta_step_len * step, 1)
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
                        s_nfeat.to(device),
                        s_adj.to(device),
                        c_nfeat.to(device)
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

                    # debug for Nan in recon loss
                    has_nan_or_inf = torch.cat(
                        [
                            torch.stack(
                                (
                                    torch.isnan(params.grad).any(),
                                    torch.isinf(params.grad).any()
                                ),
                                dim=-1
                            ) for params in model.parameters()
                        ],
                        dim=-1
                    ).any()
                    if has_nan_or_inf:
                        torch.save(
                            model,
                            path.join(save_loc, f'broken_{epoch}.ckpt')
                        )
                        torch.save(
                            s_nfeat,
                            path.join(save_loc, f's_nfeat_{epoch}.pt')
                        )
                        torch.save(
                            c_nfeat,
                            path.join(save_loc, f'c_nfeat_{epoch}.pt')
                        )
                        torch.save(
                            s_adj.to_dense(),
                            path.join(save_loc, f's_adj_{epoch}.pt')
                        )
                        torch.save(
                            seg_ids,
                            path.join(save_loc, f'seg_ids_{epoch}.pt')
                        )
                        with open(path.join(save_loc, 'batch.smi'), 'w') as f:
                            for smiles in block:
                                f.write(smiles + '\n')

                        break

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip
                    )

                    optim.step()
                    writer.add_scalar(
                        f'loss',
                        loss.cpu().item(),
                        step
                    )
                    writer.add_scalar(
                        f'recon_loss',
                        MSE.cpu().item(),
                        step
                    )
                    writer.add_scalar(
                        f'KL',
                        KL.cpu().item(),
                        step
                    )
                    step += 1
                    # print(loss.item(), MSE.item(), KL.item())
                torch.save(
                    model,
                    path.join(save_loc, f'model_{epoch}.ckpt')
                )
    except KeyboardInterrupt:
        torch.save(
            model,
            path.join(save_loc, f'model_{epoch}.ckpt')
        )

# t = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
# model = GraphInf(
#     num_in_feat=39, #     num_c_feat=4,
#     num_embeddings=num_embeddings,
#     casual_hidden_sizes=casual_hidden_sizes,
#     num_botnec_feat=num_botnec_feat,  # 16 x 4
#     num_k_feat=num_k_feat,  # 16
#     num_dense_layers=num_dense_layers,
#     num_out_feat=num_out_feat,
#     num_z_feat=num_z_feat,
#     activation=activation,
#     use_cuda=use_cuda
# )
# optim = adabound.AdaBound(
#     model.parameters(),
#     lr=LR,
#     final_lr=final_lr
# )
# model = nn.DataParallel(model, device_ids=device_ids)
# model = model.to(device)
# model.train()
# try:
#     with SummaryWriter(f'./events/{t}/') as writer:
#         for epoch in ipb(range(num_epochs), desc="epochs"):
#             if not path.exists(f'./ckpt/{t}/'):
#                 makedirs(f'./ckpt/{t}/')

#             dataloader = Dataloader(batch_size=batch_size)

#             for step, (s, c) in enumerate(
#                 ipb(
#                     dataloader,
#                     desc="step",
#                     total=dataloader.num_id_block
#                 )
#             ):

#                 num_N = s.number_of_nodes()
#                 num_E = s.number_of_edges()
#                 adj = s.adjacency_matrix().coalesce()

#                 indices = torch.cat(
#                     [adj.indices(), torch.arange(0, num_N).repeat(2, 1)],
#                     dim=-1
#                 )

#                 values = torch.ones(num_E + num_N)

#                 s_adj = torch.sparse_coo_tensor(
#                     indices,
#                     values,
#                     torch.Size([num_N, num_N])
#                 )

#                 s_nfeat = s.ndata['feat']
#                 c_nfeat = c.ndata['feat']
#                 s_nfeat, s_adj, c_nfeat = (
#                     s_nfeat.to(device), s_adj.to(device), c_nfeat.to(device)
#                 )
#                 s_nfeat = onehot_to_label(s_nfeat)
#                 c_nfeat = onehot_to_label(c_nfeat)

#                 seg_id_block = [
#                     torch.LongTensor([i]).repeat(j)
#                     for i, j in enumerate(s.batch_num_nodes)
#                 ]

#                 seg_ids = torch.cat(seg_id_block, dim=-1)

#                 x_recon, mu1, logvar1, mu2, logvar2 = (
#                     model(s_nfeat, c_nfeat, s_adj)
#                 )

#                 optim.zero_grad()
#                 MSE, KL = loss_func(
#                     x_recon, s_nfeat, mu1, logvar1, mu2, logvar2, seg_ids
#                 )
#                 loss = MSE + beta * KL
#                 loss.backward()
#                 optim.step()
#                 print(loss)
#                 writer.add_scalar(
#                     f'loss',
#                     loss.cpu().item(),
#                     step
#                 )
#                 writer.add_scalar(
#                     f'recon_loss',
#                     MSE.cpu().item(),
#                     step
#                 )
#                 writer.add_scalar(
#                     f'KL',
#                     KL.cpu().item(),
#                     step
#                 )
#                 # print(loss.item(), MSE.item(), KL.item())
#             torch.save(model, f'./ckpt/{t}/cktp_{str(epoch)}')

# except KeyboardInterrupt:
#     torch.save(model, f'./ckpt/{t}/cktp_{str(epoch)}')


def main(ckpt_loc):
    "Program entrypoint"
    with open(
        path.join(
            path.dirname(__file__),
            'ckpt',
            ckpt_loc,
            'config.json'
        )
    ) as f:
        config = json.load(f)
        config['ckpt_loc'] = ckpt_loc
        engine(**config)


if __name__ == '__main__':
    main(sys.argv[1])
