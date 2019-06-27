import os.path as op
from multiprocessing import cpu_count
# import random

from joblib import Parallel, delayed
import dgl

from utils import *

__all__ = [
    'Dataloader',
]


class Dataloader(object):
    def __init__(
        self,
        original_scaffolds_file: str=op.join(
            op.dirname(__file__),
            'data-center',
            'scaffolds_a.smi'
        ),
        c_scaffolds_file: str=op.join(
            op.dirname(__file__),
            'data-center',
            'scaffolds_c.smi'
        ),
        batch_size: int=400,
        collate_fn: str='dgl',
        num_workers: int=cpu_count()
    ):
        super().__init__()
        self.o_scaffolds = original_scaffolds_file
        self.c_scaffolds = c_scaffolds_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_line = get_num_lines(self.o_scaffolds)
        self.smiles_blocks = str_block_gen(
            self.o_scaffolds,
            self.batch_size
        )
        self.num_id_block = len(self.smiles_blocks)
        # # self.shuffled_id = list(range(self.num_line))
        # random.shuffle(self.shuffled_id)
        # self.num_id_block = (
        #     self.num_line // self.batch_size if
        #     self.num_line % self.batch_size == 0 else
        #     self.num_line // self.batch_size + 1
        # )
        # self.id_block = [
        #     self.shuffled_id[
        #         i * self.batch_size:min(
        #             (i + 1) * self.batch_size, self.num_line
        #         )
        #     ]
        #     for i in range(self.num_id_block)
        # ]

    def __len__(self):
        assert (
            get_num_lines(self.o_scaffolds) ==
            get_num_lines(self.c_scaffolds)
        )
        return self.num_id_block

    def __iter__(self):
        # smiles_blocks = str_block_gen(
        #     self.o_scaffolds,
        #     self.batch_size
        # )
        # p = Pool(self.num_workers)

        for block in self.smiles_blocks:
            # p = Pool(self.num_workers)
            # ls_scaffold = p.map(
            #     whole_graph_from_smiles,
            #     block,
            #     chunksize=1000
            # )

            ls_scaffold = Parallel(
                n_jobs=self.num_workers,
                backend='multiprocessing'
            )(
                delayed(whole_graph_from_smiles)
                (
                    i
                )
                for i in block
            )

            ls_o_scaffold_clean = [
                s_pair[0] for s_pair in ls_scaffold if s_pair[0] is not None
            ]
            ls_c_scaffold_clean = [
                s_pair[1] for s_pair in ls_scaffold if s_pair[1] is not None
            ]

            yield (
                dgl.batch(ls_o_scaffold_clean),
                dgl.batch(ls_c_scaffold_clean)
            )

        # for block in self.id_block:
        #     ls_o_scaffold = Parallel(
        #         n_jobs=self.num_workers,
        #         backend='multiprocessing'
        #     )(
        #         delayed(whole_graph_from_line)
        #         (
        #             self.o_scaffolds,
        #             i
        #         )
        #         for i in block
        #     )

        #     ls_c_scaffold = Parallel(
        #         n_jobs=self.num_workers,
        #         backend='multiprocessing'
        #     )(
        #         delayed(whole_graph_from_line)
        #         (
        #             self.o_scaffolds,
        #             i,
        #             True
        #         )
        #         for i in block
        #     )

        #     ls_o_scaffold_clean = [_ for _ in ls_o_scaffold if _ is not None]
        #     ls_c_scaffold_clean = [_ for _ in ls_c_scaffold if _ is not None]

        #     yield (
        #         dgl.batch(ls_o_scaffold_clean),
        #         dgl.batch(ls_c_scaffold_clean)
        #     )
            # yield block
