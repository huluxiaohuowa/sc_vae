import os.path as op
from multiprocessing import cpu_count
import random

import torch
from joblib import Parallel
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
            'a_scaffolds.smi'
        ),
        c_scaffolds_file: str=op.join(
            op.dirname(__file__),
            'data-center',
            'c_scaffolds_file'
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
        self.shuffled_id = list(range(self.num_line))
        random.shuffle(self.shuffled_id)
        self.num_id_block = (
            self.num_line // self.batch_size if 
            self.num_line % self.batch_size == 0 else 
            self.num_line // self.batch_size + 1
        )
        self.id_block = [
            self.shuffled_id[
                i * self.batch_size:min((i + 1) * self.batch_size, self.num_line)
            ]
            for i in range(self.num_id_block)
        ]

    def __len__(self):
        assert (
            get_num_lines(self.o_scaffolds) == 
            get_num_lines(self.c_scaffolds)
        )
        return self.num_lines

    def __iter__():
        for block in self.id_block:
            ls_o_scaffold = Parallel(
                n_jobs=self.num_workers, 
                backend='multiprocessing')
            (
                delayed(graph_from_line)
                (
                    self.o_scaffolds,
                    i
                )
                for i in block
            )
            ls_c_scaffold = Parallel(
                n_jobs=self.num_workers, 
                backend='multiprocessing')
            (
                delayed(graph_from_line)
                (
                    self.c_scaffolds,
                    i,
                    True
                )
                for i in block
            )

            yield dgl.batch(ls_o_scaffold), dgl.batch(ls_c_scaffold)




