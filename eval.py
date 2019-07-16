import sys
from os import path
import json

import torch
from ipypb import ipb
# import numpy
import multiprocess as mp
from joblib import Parallel, delayed

from networks import *
from data import *
from utils import *
from mol_spec import *

# config_id = 'naive3'
# device_id = 2
# scaffolds_file = 'data-center/scaffolds_a.smi'

ms = MoleculeSpec.get_default()


def engine(
    config_id='naive3',
    device_id=2,
    model_idx=4,
    scaffolds_file='data-center/test.smi',
    batch_size=500,
    np=mp.cpu_count(),
):
    device = torch.device(f'cuda:{device_id}')
    model_ckpt = path.join(
        path.dirname(__file__),
        'ckpt',
        config_id,
        f'model_{model_idx}.ckpt'
    )
    # print(model_ckpt)
    model = torch.load(model_ckpt).to(device)
    model.eval()

    dataloader = ComLoader(
        original_scaffolds_file=scaffolds_file,
        batch_size=batch_size,
        num_workers=1
    )

    all_num_valid = 0
    all_num_recon = 0

    with open(f'eval_configs/{config_id}_records.txt', 'w') as f:
        for batch in ipb(
            dataloader,
            desc="step",
            total=dataloader.num_id_block
        ):
            (
                block,
                nums_nodes,
                nums_edges,
                seg_ids,
                bond_info_all,
                nodes_o,
                nodes_c
            ) = batch
            num_N = sum(nums_nodes)
            num_E = sum(nums_edges)

            values = torch.ones(num_E)

            s_adj = torch.sparse_coo_tensor(
                bond_info_all.T,
                values,
                torch.Size([num_N, num_N])
            ).to(device)

            s_nfeat = torch.from_numpy(nodes_o).to(device)
            c_nfeat = torch.from_numpy(nodes_c).to(device)

            x_inf = model.inf(c_nfeat, s_adj).cpu().detach()
            x_recon = model.reconstrcut(s_nfeat, s_adj).cpu().detach()

            ls_x_inf = torch.split(x_inf, nums_nodes)
            ls_x_recon = torch.split(x_recon, nums_nodes)
            ls_mols_inf = Parallel(
                n_jobs=np,
                backend='multiprocessing'
            )(
                delayed(get_mol_from_array)
                (
                    ls_x_inf[i], block[i], True, False
                )
                for i in range(len(block))
            )
            ls_mols_recon = Parallel(
                n_jobs=np,
                backend='multiprocessing'
            )(
                delayed(get_mol_from_array)
                (
                    ls_x_recon[i], block[i], True, True
                )
                for i in range(len(block))
            )
            num_valid = sum(x is not None for x in ls_mols_inf)
            num_recon = sum(
                ls_mols_recon[i] == block[i] for i in range(len(block))
            )
            all_num_valid += num_valid
            all_num_recon += num_recon
            f.write(
                str(num_valid) + '\t' +
                str(num_recon) + '\t' +
                str(len(ls_mols_inf)) + '\n'
            )
            f.flush()
    with open(f'eval_configs/{config_id}.txt', 'w') as f:
        f.write(str(all_num_valid) + '\t')
        f.write(str(all_num_recon))

    # for i, (s, c, block) in ipb(
    #     enumerate(dataloader.legacy_iter(mode=mode)),
    #     total=dataloader.num_id_block
    # ):
    #     num_N = s.number_of_nodes()
    #     num_E = s.number_of_edges()
    #     adj = s.adjacency_matrix().coalesce()
    #     indices = torch.cat(
    #         [adj.indices(), torch.arange(0, num_N).repeat(2, 1)],
    #         dim=-1
    #     )
    #     values = torch.ones(num_E + num_N)
    #     s_adj = torch.sparse_coo_tensor(
    #         indices,
    #         values,
    #         torch.Size([num_N, num_N])
    #     )
    #     s_nfeat = s.ndata['feat']
    #     c_nfeat = c.ndata['feat']
    #     s_nfeat, s_adj, c_nfeat = (
    #         s_nfeat.to(device),
    #         s_adj.to(device),
    #         c_nfeat.to(device)
    #     )
    #     s_nfeat = onehot_to_label(s_nfeat)
    #     c_nfeat = onehot_to_label(c_nfeat)

    #     # seg_id_block = [
    #     #     torch.LongTensor([i]).repeat(j)
    #     #     for i, j in enumerate(s.batch_num_nodes)
    #     # ]

    #     # seg_ids = torch.cat(seg_id_block, dim=-1)

    #     x_inf = model.inf(c_nfeat, s_adj).cpu().detach()
    #     x_recon = model.reconstrcut(s_nfeat, s_adj).cpu().detach()

    #     ls_x_inf = torch.split(x_inf, s.batch_num_nodes)
    #     ls_x_recon = torch.split(x_recon, s.batch_num_nodes)

    #     ls_mols_inf = Parallel(
    #         n_jobs=np,
    #         backend='multiprocessing'
    #     )(
    #         delayed(get_mol_from_array)
    #         (
    #             ls_x_inf[i], block[i], True, False
    #         )
    #         for i in range(len(block))
    #     )
    #     ls_mols_recon = Parallel(
    #         n_jobs=np,
    #         backend='multiprocessing'
    #     )(
    #         delayed(get_mol_from_array)
    #         (
    #             ls_x_recon[i], block[i], True, True
    #         )
    #         for i in range(len(block))
    #     )

    #     num_valid = sum(x is not None for x in ls_mols_inf)
    #     num_recon = sum(
    #         ls_mols_recon[i] == block[i] for i in range(len(block))
    #     )
    #     all_num_valid += num_valid
    #     all_num_recon += num_recon

    # with open(f'eval_configs/{config_id}.txt', 'w') as f:
    #     f.write(str(all_num_valid) + '\t')
    #     f.write(str(all_num_recon))


def main(config_id):
    "Program entrypoint"
    with open(
        path.join(
            path.dirname(__file__),
            'eval_configs',
            f'{config_id}.json',
        )
    ) as f:
        config = json.load(f)
        config['config_id'] = config_id
        engine(**config)


if __name__ == '__main__':
    main(sys.argv[1])
