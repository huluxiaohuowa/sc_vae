import os.path as op
from multiprocessing import cpu_count
# import typing as t
from threading import Thread
import abc

# from joblib import Parallel, delayed
import multiprocess as mp
# import multiprocessing as mp
import dgl
import numpy as np

from utils import *
from graph import *

__all__ = [
    'Dataloader',
    'ComLoader'
]


class Loader(object):
    def __init__(
        self,
        original_scaffolds_file: str=op.join(
            op.dirname(__file__),
            'data-center',
            'scaffolds_a.smi'
        ),
        # c_scaffolds_file: str=op.join(
        #     op.dirname(__file__),
        #     'data-center',
        #     'scaffolds_c.smi'
        # ),
        batch_size: int=400,
        num_workers: int=cpu_count()
    ):
        super().__init__()
        self.o_scaffolds = original_scaffolds_file
        # self.c_scaffolds = c_scaffolds_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_line = get_num_lines(self.o_scaffolds)
        self.smiles_blocks = str_block_gen(
            self.o_scaffolds,
            self.batch_size
        )
        self.num_id_block = len(self.smiles_blocks)

        # self.num_train_blocks = self.num_id_block // 10 * 8
        # self.num_test_blocks = self.num_id_block - self.num_train_blocks

        # self.train_blocks = self.smiles_blocks[:self.num_train_blocks]
        # self.test_blocks = self.smiles_blocks[self.num_train_blocks:]

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    # @property
    # @abc.abstractmethod
    # def train(self):
    #     raise NotImplementedError

    # @property
    # @abc.abstractmethod
    # def test(self):
    #     raise NotImplementedError


class ComLoader(Loader):
    def __init__(self, **kargs):
        super(ComLoader, self).__init__(**kargs)

    def __len__():
        return self.num_id_block

    def __iter__(self):
        queue_in, queue_out = (
            mp.Queue(self.num_workers * 3),
            mp.Queue(self.num_workers * 3)
        )

        def _worker_g():
            for smiles_block_i in self.smiles_blocks:
                queue_in.put(smiles_block_i)
            for _ in range(self.num_workers):
                queue_in.put(None)

        def _workder():
            while True:
                _block = queue_in.get()
                if _block is None:
                    break
                nums_nodes = []
                nums_edges = []
                bond_infos = []
                nodes_os = []
                nodes_cs = []
                for smiles_i in _block:
                    try:
                        weave_mol = WeaveMol(smiles_i)
                        num_nodes = weave_mol.num_all_nodes
                        num_edges = weave_mol.num_all_edges
                        all_edges = weave_mol.all_edges
                        all_nodes_o = weave_mol.all_nodes_o
                        all_nodes_c = weave_mol.all_nodes_c

                        nums_nodes.append(num_nodes)
                        nums_edges.append(num_edges)
                        bond_infos.append(all_edges)
                        nodes_os.append(all_nodes_o)
                        nodes_cs.append(all_nodes_c)
                    except:
                        pass
                queue_out.put((
                    _block,
                    nums_nodes,
                    nums_edges,
                    bond_infos,
                    nodes_os,
                    nodes_cs
                ))
            queue_out.put(None)

        thread = Thread(target=_worker_g)
        thread.start()

        pool = [mp.Process(target=_workder) for _ in range(self.num_workers)]
        for p in pool:
            p.start()

        exit_workers = 0
        while exit_workers < self.num_workers:
            record = queue_out.get()
            if record is None:
                exit_workers += 1
                continue
            (
                block,
                nums_nodes,
                nums_edges,
                bond_infos,
                nodes_os,
                nodes_cs
            ) = record

            seg_ids = np.arange(len(nums_nodes)).repeat(nums_nodes)

            bond_info = np.concatenate(bond_infos, axis=0)
            shift_values = np.pad(
                np.cumsum(nums_nodes),
                1,
                'constant',
                constant_values=0
            )[: -2].repeat(nums_edges).reshape([-1, 1])

            bond_info_all = bond_info + shift_values

            nodes_o = np.concatenate(nodes_os, axis=-1)
            nodes_c = np.concatenate(nodes_cs, axis=-1)

            yield (
                block,
                nums_nodes,
                nums_edges,
                seg_ids,
                bond_info_all,
                nodes_o,
                nodes_c
            )

    # @property
    # def train(self):
    #     queue_in, queue_out = (
    #         mp.Queue(self.num_workers * 3),
    #         mp.Queue(self.num_workers * 3)
    #     )

    #     def _worker_g():
    #         for smiles_block_i in self.train_blocks:
    #             queue_in.put(smiles_block_i)
    #         for _ in range(self.num_workers):
    #             queue_in.put(None)

    #     def _workder():
    #         while True:
    #             _block = queue_in.get()
    #             if _block is None:
    #                 break
    #             nums_nodes = []
    #             nums_edges = []
    #             bond_infos = []
    #             nodes_os = []
    #             nodes_cs = []
    #             for smiles_i in _block:
    #                 try:
    #                     weave_mol = WeaveMol(smiles_i)
    #                     num_nodes = weave_mol.num_all_nodes
    #                     num_edges = weave_mol.num_all_edges
    #                     all_edges = weave_mol.all_edges
    #                     all_nodes_o = weave_mol.all_nodes_o
    #                     all_nodes_c = weave_mol.all_nodes_c

    #                     nums_nodes.append(num_nodes)
    #                     nums_edges.append(num_edges)
    #                     bond_infos.append(all_edges)
    #                     nodes_os.append(all_nodes_o)
    #                     nodes_cs.append(all_nodes_c)
    #                 except:
    #                     pass
    #             queue_out.put((
    #                 _block,
    #                 nums_nodes,
    #                 nums_edges,
    #                 bond_infos,
    #                 nodes_os,
    #                 nodes_cs
    #             ))
    #         queue_out.put(None)

    #     thread = Thread(target=_worker_g)
    #     thread.start()

    #     pool = [mp.Process(target=_workder) for _ in range(self.num_workers)]
    #     for p in pool:
    #         p.start()

    #     exit_workers = 0
    #     while exit_workers < self.num_workers:
    #         record = queue_out.get()
    #         if record is None:
    #             exit_workers += 1
    #             continue
    #         (
    #             block,
    #             nums_nodes,
    #             nums_edges,
    #             bond_infos,
    #             nodes_os,
    #             nodes_cs
    #         ) = record

    #         seg_ids = np.arange(len(nums_nodes)).repeat(nums_nodes)

    #         bond_info = np.concatenate(bond_infos, axis=0)
    #         shift_values = np.pad(
    #             np.cumsum(nums_nodes),
    #             1,
    #             'constant',
    #             constant_values=0
    #         )[: -2].repeat(nums_edges).reshape([-1, 1])

    #         bond_info_all = bond_info + shift_values

    #         nodes_o = np.concatenate(nodes_os, axis=-1)
    #         nodes_c = np.concatenate(nodes_cs, axis=-1)

    #         yield (
    #             block,
    #             nums_nodes,
    #             nums_edges,
    #             seg_ids,
    #             bond_info_all,
    #             nodes_o,
    #             nodes_c
    #         )

    # @property
    # def test(self):
    #     queue_in, queue_out = (
    #         mp.Queue(self.num_workers * 3),
    #         mp.Queue(self.num_workers * 3)
    #     )

    #     def _worker_g():
    #         for smiles_block_i in self.test_blocks:
    #             queue_in.put(smiles_block_i)
    #         for _ in range(self.num_workers):
    #             queue_in.put(None)

    #     def _workder():
    #         while True:
    #             _block = queue_in.get()
    #             if _block is None:
    #                 break
    #             nums_nodes = []
    #             nums_edges = []
    #             bond_infos = []
    #             nodes_os = []
    #             nodes_cs = []
    #             for smiles_i in _block:
    #                 try:
    #                     weave_mol = WeaveMol(smiles_i)
    #                     num_nodes = weave_mol.num_all_nodes
    #                     num_edges = weave_mol.num_all_edges
    #                     all_edges = weave_mol.all_edges
    #                     all_nodes_o = weave_mol.all_nodes_o
    #                     all_nodes_c = weave_mol.all_nodes_c

    #                     nums_nodes.append(num_nodes)
    #                     nums_edges.append(num_edges)
    #                     bond_infos.append(all_edges)
    #                     nodes_os.append(all_nodes_o)
    #                     nodes_cs.append(all_nodes_c)
    #                 except:
    #                     pass
    #             queue_out.put((
    #                 _block,
    #                 nums_nodes,
    #                 nums_edges,
    #                 bond_infos,
    #                 nodes_os,
    #                 nodes_cs
    #             ))
    #         queue_out.put(None)

    #     thread = Thread(target=_worker_g)
    #     thread.start()

    #     pool = [mp.Process(target=_workder) for _ in range(self.num_workers)]
    #     for p in pool:
    #         p.start()

    #     exit_workers = 0
    #     while exit_workers < self.num_workers:
    #         record = queue_out.get()
    #         if record is None:
    #             exit_workers += 1
    #             continue
    #         (
    #             block,
    #             nums_nodes,
    #             nums_edges,
    #             bond_infos,
    #             nodes_os,
    #             nodes_cs
    #         ) = record

    #         seg_ids = np.arange(len(nums_nodes)).repeat(nums_nodes)

    #         bond_info = np.concatenate(bond_infos, axis=0)
    #         shift_values = np.pad(
    #             np.cumsum(nums_nodes),
    #             1,
    #             'constant',
    #             constant_values=0
    #         )[: -2].repeat(nums_edges).reshape([-1, 1])

    #         bond_info_all = bond_info + shift_values

    #         nodes_o = np.concatenate(nodes_os, axis=-1)
    #         nodes_c = np.concatenate(nodes_cs, axis=-1)

    #         yield (
    #             block,
    #             nums_nodes,
    #             nums_edges,
    #             seg_ids,
    #             bond_info_all,
    #             nodes_o,
    #             nodes_c
    #         )


class Dataloader(Loader):
    def __init__(
        self,
        **kargs
    ):
        super(Dataloader, self).__init__(**kargs)

    def __len__(self):
        # assert (
        #     get_num_lines(self.o_scaffolds) ==
        #     get_num_lines(self.c_scaffolds)
        # )
        return self.num_id_block

    def __iter__(self):
        # smiles_blocks = str_block_gen(
        #     self.o_scaffolds,
        #     self.batch_size
        # )
        # p = Pool(self.num_workers)
        queue_in, queue_out = (mp.Queue(self.num_workers * 3),
                               mp.Queue(self.num_workers * 3))

        def _worker_g():
            for smiles_block_i in self.smiles_blocks:
                queue_in.put(smiles_block_i)
            for _ in range(self.num_workers):
                queue_in.put(None)

        def _worker():
            while True:
                _block = queue_in.get()
                if _block is None:
                    break
                results = []
                for smiles_i in _block:
                    result_i = whole_graph_from_smiles(smiles_i)
                    results.append(result_i)
                queue_out.put((results, _block))
            queue_out.put(None)

        thread = Thread(target=_worker_g)
        thread.start()

        pool = [mp.Process(target=_worker) for _ in range(self.num_workers)]
        for p in pool:
            p.start()

        exit_workers = 0
        while exit_workers < self.num_workers:
            record = queue_out.get()
            if record is None:
                exit_workers += 1
                continue
            ls_scaffold, block = record
            ls_o_scaffold_clean = [
                s_pair[0] for s_pair in ls_scaffold if s_pair[0] is not None
            ]
            ls_c_scaffold_clean = [
                s_pair[1] for s_pair in ls_scaffold if s_pair[1] is not None
            ]

            yield (
                dgl.batch(ls_o_scaffold_clean),
                dgl.batch(ls_c_scaffold_clean),
                block
            )

    # @property
    # def train(
    #     self
    # ):
    #     queue_in, queue_out = (mp.Queue(self.num_workers * 3),
    #                            mp.Queue(self.num_workers * 3))

    #     def _worker_g():
    #         for smiles_block_i in self.train_blocks:
    #             queue_in.put(smiles_block_i)
    #         for _ in range(self.num_workers):
    #             queue_in.put(None)

    #     def _worker():
    #         while True:
    #             _block = queue_in.get()
    #             if _block is None:
    #                 break
    #             results = []
    #             for smiles_i in _block:
    #                 result_i = whole_graph_from_smiles(smiles_i)
    #                 results.append(result_i)
    #             queue_out.put((results, _block))
    #         queue_out.put(None)

    #     thread = Thread(target=_worker_g)
    #     thread.start()

    #     pool = [mp.Process(target=_worker) for _ in range(self.num_workers)]
    #     for p in pool:
    #         p.start()

    #     exit_workers = 0
    #     while exit_workers < self.num_workers:
    #         record = queue_out.get()
    #         if record is None:
    #             exit_workers += 1
    #             continue
    #         ls_scaffold, block = record
    #         ls_o_scaffold_clean = [
    #             s_pair[0] for s_pair in ls_scaffold if s_pair[0] is not None
    #         ]
    #         ls_c_scaffold_clean = [
    #             s_pair[1] for s_pair in ls_scaffold if s_pair[1] is not None
    #         ]

    #         yield (
    #             dgl.batch(ls_o_scaffold_clean),
    #             dgl.batch(ls_c_scaffold_clean),
    #             block
    #         )

    # @property
    # def test(
    #     self
    # ):
    #     queue_in, queue_out = (mp.Queue(self.num_workers * 3),
    #                            mp.Queue(self.num_workers * 3))

    #     def _worker_g():
    #         for smiles_block_i in self.test_blocks:
    #             queue_in.put(smiles_block_i)
    #         for _ in range(self.num_workers):
    #             queue_in.put(None)

    #     def _worker():
    #         while True:
    #             _block = queue_in.get()
    #             if _block is None:
    #                 break
    #             results = []
    #             for smiles_i in _block:
    #                 result_i = whole_graph_from_smiles(smiles_i)
    #                 results.append(result_i)
    #             queue_out.put((results, _block))
    #         queue_out.put(None)

    #     thread = Thread(target=_worker_g)
    #     thread.start()

    #     pool = [mp.Process(target=_worker) for _ in range(self.num_workers)]
    #     for p in pool:
    #         p.start()

    #     exit_workers = 0
    #     while exit_workers < self.num_workers:
    #         record = queue_out.get()
    #         if record is None:
    #             exit_workers += 1
    #             continue
    #         ls_scaffold, block = record
    #         ls_o_scaffold_clean = [
    #             s_pair[0] for s_pair in ls_scaffold if s_pair[0] is not None
    #         ]
    #         ls_c_scaffold_clean = [
    #             s_pair[1] for s_pair in ls_scaffold if s_pair[1] is not None
    #         ]

    #         yield (
    #             dgl.batch(ls_o_scaffold_clean),
    #             dgl.batch(ls_c_scaffold_clean),
    #             block
    #         )

    # def legacy_iter(
    #     self,
    #     mode: str="all"
    # ):
    #     dic_mode = {
    #         "all": self.smiles_blocks,
    #         "train": self.train_blocks,
    #         "test": self.test_blocks
    #     }
    #     for block in dic_mode[mode]:
    #         ls_scaffold = Parallel(
    #             n_jobs=mp.cpu_count(),
    #             backend='multiprocessing'
    #         )(
    #             delayed(whole_graph_from_smiles)
    #             (
    #                 i
    #             )
    #             for i in block
    #         )

    #         ls_o_scaffold_clean = [
    #             s_pair[0] for s_pair in ls_scaffold if s_pair[0] is not None
    #         ]
    #         ls_c_scaffold_clean = [
    #             s_pair[1] for s_pair in ls_scaffold if s_pair[1] is not None
    #         ]

    #         yield (
    #             dgl.batch(ls_o_scaffold_clean),
    #             dgl.batch(ls_c_scaffold_clean),
    #             block
    #         )
