import typing as t
from copy import deepcopy
from itertools import chain

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops
from scipy import sparse

from utils import *
from mol_spec import MoleculeSpec


__all__ = [
    # "smiles_to_nx_graph",
    "WeaveMol",
]

ms = MoleculeSpec.get_default()


# def smiles_to_nx_graph(smiles: str) -> nx.classes.graph.Graph:
#     mol = Chem.MolFromSmiles(smiles)
#     atom_types, bonds, bond_types = [], [], []
#     for atom in mol.GetAtoms():
#         atom_types.append(ms.get_atom_type(atom))
#     for bond in mol.GetBonds():
#         idx_1, idx_2, bond_type = (
#             bond.GetBeginAtomIdx(),
#             bond.GetEndAtomIdx(),
#             ms.get_bond_type(bond)
#         )
#         bonds.append([idx_1, idx_2])
#         bond_types.append(bond_type)
#     # build graph
#     graph = nx.Graph()
#     graph.add_nodes_from(range(mol.GetNumAtoms()))
#     graph.add_edges_from(bonds)
#     return graph


class WeaveMol(object):
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.ms = ms
        self.num_atom_types = ms.num_atom_types
        self.num_bond_types = ms.num_bond_types

    @property
    def num_atoms(self):
        return self.mol.GetNumAtoms()

    @property
    def original_atoms(self):
        return list(range(self.num_atoms))

    @property
    def num_original_bonds(self):
        return self.mol.GetNumBonds()

    @property
    def atom_types(self):
        atom_types = [ms.get_atom_type(atom) for atom in self.mol.GetAtoms()]
        return np.array(atom_types, dtype=np.int)

    @property
    def atom_type_c(self):
        return np.zeros_like(self.atom_types, dtype=np.int)

    @property
    def original_bond_info(self):
        bond_info = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in self.mol.GetBonds()
        ]
        return bond_info  # N_bond x 2

    # @property
    # def original_bond_types(self):
    #     bond_types = [
    #         ms.get_bond_type(bond.GetBondType())
    #         for bond in self.mol.GetBonds()
    #     ]
    #     return bond_types

    @property
    def original_bond_types(self, to_array=False):
        bond_types = [
            ms.get_bond_type(bond) for bond in self.mol.GetBonds()
        ]
        return np.array(bond_types, dtype=np.int)

    @property
    def original_bond_types_c(self):
        return np.zeros_like(self.original_bond_types, dtype=np.int)

    @property
    def original_graph(self) -> nx.classes.graph.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_atoms))
        graph.add_edges_from(self.original_bond_info)
        return graph

    @property
    def original_adj(self):
        indices = self.original_bond_info_np.repeat(2, 0)  # 2N_bonds x 2
        row, col, data = (
            indices[:, 0],
            indices[:, 1],
            np.ones([indices.shape[0], ], dtype=np.float32)
        )
        adj = sparse.coo_matrix(
            (data, (row, col)),
            shape=(self.num_atoms, self.num_atoms)
        )
        return adj

    @property
    def original_bond_info_np(self):
        return np.stack(self.original_bond_info)

    @property
    def remote_connections(self):
        d = self.original_adj * self.original_adj
        d_indices_2 = np.stack(d.nonzero(), axis=1)
        # remove diagonal elements
        d_indices_2 = d_indices_2[d_indices_2[:, 0] != d_indices_2[:, 1], :]

        d = d * self.original_adj
        d = d - d.multiply(self.original_adj)

        d_indices_3 = np.stack(d.nonzero(), axis=1)
        # remove diagonal elements
        d_indices_3 = d_indices_3[d_indices_3[:, 0] != d_indices_3[:, 1], :]

        return d_indices_2, d_indices_3  # N_bond x 2

    @property
    def remote_connection_2(self):
        return self.remote_connections[0]

    @property
    def remote_connection_3(self):
        return self.remote_connections[1]

    @property
    def num_remote_connection_2(self):
        return self.remote_connections[0].shape[0]

    @property
    def num_remote_connection_3(self):
        return self.remote_connections[1].shape[0]

    @property
    def sssr(self) -> t.List[t.List]:
        """Get sssr atom indices of a molecule
        
        Returns:
            t.List[t.List]: [[sssr1 atomindices],[sssr2 atom indices], ...]
        """
        return [list(ring) for ring in rdmolops.GetSymmSSSR(self.mol)]

    @property
    def num_original_rings(self) -> int:
        return len(self.sssr)

    @property
    def new_atoms(self):
        return (
            list(
                range(
                    self.num_atoms,
                    self.num_atoms + self.num_original_bonds
                )
            )
        )

    @property
    def new_nodes(self):
        return self.original_atoms + self.new_atoms

    @property
    def new_bond_info_np(self):
        new_bond_info_np = np.concatenate(
            [
                self.original_bond_info_np,
                np.array(
                    self.new_atoms,
                    dtype=np.int
                ).reshape([-1, 1])
            ],
            axis=1
        )
        return new_bond_info_np

    @property
    def new_bond_info_dict(self):
        new_bond_info_dict = {}
        for new_bond in self.new_bond_info_np:
            new_bond_info_dict[(new_bond[0], new_bond[1])] = new_bond[2]
        return new_bond_info_dict

    @property
    def new_edge_info(self):
        bonds1 = np.concatenate(
            [
                self.new_bond_info_np[:, 0].reshape([-1, 1]),
                self.new_bond_info_np[:, 2].reshape([-1, 1])
            ], axis=-1
        )
        bonds2 = np.concatenate(
            [
                self.new_bond_info_np[:, 2].reshape([-1, 1]),
                self.new_bond_info_np[:, 1].reshape([-1, 1])
            ], axis=-1
        )
        bonds = np.concatenate([bonds1, bonds2], axis=0)
        return bonds

    @property
    def original_ring_bond_info(self):
        ring_bond_info = []
        ring_bond_info_set = []
        for ring in self.sssr:
            ring_bond_set = []
            num_ring_atoms = len(ring)
            for idx_atom in range(num_ring_atoms):
                if idx_atom >= num_ring_atoms - 1:
                    if (ring[idx_atom], ring[0]) in self.original_bond_info:
                        bond_info = (ring[idx_atom], ring[0])
                    else:
                        bond_info = (ring[0], ring[idx_atom])
                    ring_bond_info.append(bond_info)
                    ring_bond_set.append(bond_info)
                else:
                    if (
                        (ring[idx_atom], ring[idx_atom + 1])
                        in self.original_bond_info
                    ):
                        bond_info = (ring[idx_atom], ring[idx_atom + 1])
                    else:
                        bond_info = (ring[idx_atom + 1], ring[idx_atom])
                    ring_bond_info.append(bond_info)
                    ring_bond_set.append(bond_info)
            ring_bond_info_set.append(ring_bond_set)
        return list(set(ring_bond_info)), ring_bond_info_set

    @property
    def original_chain_bond_info(self):
        return (
            list(
                set(self.original_bond_info) -
                set(self.original_ring_bond_info[0])
            )
        )

    @property
    def new_sssr(self):
        new_sssr = deepcopy(self.sssr)
        for i, sssr in enumerate(new_sssr):
            for bond_info in self.original_ring_bond_info[1][i]:
                sssr.append(self.new_bond_info_dict[bond_info])
        return new_sssr

    @property
    def new_ring_atoms(self):
        # atoms = list(set([i for j in self.new_sssr for i in j]))
        atoms = list(set(chain(*self.new_sssr)))
        return atoms

    @property
    def new_graph(self):
        g = nx.Graph()
        g.add_nodes_from(self.new_nodes)
        g.add_edges_from(self.new_edge_info)
        return g

    @property
    def new_chain_nodes(self):
        return(
            list(
                set(self.new_nodes) - set(self.new_ring_atoms)
            )
        )

    @property
    def ring_assems(self):
        g = deepcopy(self.new_graph)
        g.remove_nodes_from(self.new_chain_nodes)
        ring_assems = list(nx.connected_component_subgraphs(g))
        ring_assems_nodes = [list(graph.nodes()) for graph in ring_assems]
        return ring_assems_nodes

    @property
    def chain_assems(self):
        g = deepcopy(self.new_graph)
        g.remove_nodes_from(self.new_ring_atoms)
        chain_assems = list(nx.connected_component_subgraphs(g))
        chain_assems_nodes = [list(graph.nodes()) for graph in chain_assems]
        return chain_assems_nodes

    @property
    def remote_connection_bonds(self):
        num_new_nodes = len(self.new_nodes)
        remote_nodes_2 = list(range(
            num_new_nodes,
            num_new_nodes + self.num_remote_connection_2
        ))
        remote_nodes_3 = list(range(
            num_new_nodes + self.num_remote_connection_2,
            num_new_nodes + self.num_remote_connection_2 +
            self.num_remote_connection_3
        ))
        remote_bonds2 = np.concatenate([
            np.concatenate([
                self.remote_connection_2[:, 0].reshape([-1, 1]),
                np.array(remote_nodes_2, dtype=np.int).reshape([-1, 1])
            ], axis=-1),
            np.concatenate([
                np.array(remote_nodes_2, dtype=np.int).reshape([-1, 1]),
                self.remote_connection_2[:, 1].reshape([-1, 1])
            ], axis=-1)
        ], axis=0)
        remote_bonds3 = np.concatenate([
            np.concatenate([
                self.remote_connection_3[:, 0].reshape([-1, 1]),
                np.array(remote_nodes_3, dtype=np.int).reshape([-1, 1])
            ], axis=-1),
            np.concatenate([
                np.array(remote_nodes_3, dtype=np.int).reshape([-1, 1]),
                self.remote_connection_3[:, 1].reshape([-1, 1])
            ], axis=-1)
        ], axis=0)
        return remote_bonds2, remote_bonds3

    @property
    def remote_bonds_2(self):
        return self.remote_connection_bonds[0]

    @property
    def remote_bonds_3(self):
        return self.remote_connection_bonds[1]

    @property
    def chains_master_nodes(self):
        num_above = (
            self.num_atoms +
            self.num_original_bonds +
            self.num_remote_connection_2 +
            self.num_remote_connection_3
        )
        chains_master_nodes = list(
            range(
                num_above, num_above + len(self.chain_assems)
            )
        )
        return chains_master_nodes

    @property
    def num_chains_master_nodes(self):
        return len(self.chains_master_nodes)

    @property
    def chains_master_edges(self):
        num_atoms_each_chain = [len(chain) for chain in self.chain_assems]
        master_nodes = np.repeat(
            self.chains_master_nodes,
            num_atoms_each_chain,
            axis=0
        )
        all_chain_nodes = list(chain(*self.chain_assems))
        chains_master_edges = np.stack(
            [master_nodes, all_chain_nodes], axis=0
        ).T
        return chains_master_edges
        # return master_nodes, all_chain_nodes

        # for i, chain_assem in enumerate(self.chain_assems):
        #     for j in chain_assem:
        #         chains_master_edges.append([self.chains_master_nodes[i], j])
        # chains_master_edges = np.stack(chains_master_edges, axis=0)
        # return chains_master_edges

    @property
    def rings_master_nodes(self):
        num_above = (
            self.num_atoms +
            self.num_original_bonds +
            self.num_remote_connection_2 +
            self.num_remote_connection_3 +
            self.num_chains_master_nodes
        )
        rings_master_nodes = list(
            range(
                num_above, num_above + self.num_original_rings
            )
        )
        return rings_master_nodes

    @property
    def num_rings_master_nodes(self):
        return len(self.rings_master_nodes)

    @property
    def rings_master_edges(self):
        # rings_master_edges = []
        # for i, ring in enumerate(self.new_sssr):
        #     for j in ring:
        #         rings_master_edges.append([self.rings_master_nodes[i], j])
        # rings_master_edges = np.stack(rings_master_edges, axis=0)
        # return rings_master_edges
        num_atoms_each_ring = [len(ring) for ring in self.new_sssr]
        master_nodes = np.repeat(
            self.rings_master_nodes,
            num_atoms_each_ring,
            axis=0
        )
        all_ring_nodes = list(chain(*self.new_sssr))
        rings_master_edges = np.stack(
            [master_nodes, all_ring_nodes], axis=0
        ).T
        return rings_master_edges

    @property
    def ringassems_master_nodes(self):
        num_above = (
            self.num_atoms +
            self.num_original_bonds +
            self.num_remote_connection_2 +
            self.num_remote_connection_3 +
            self.num_chains_master_nodes +
            self.num_original_rings
        )
        ringassesm_master_nodes = list(
            range(
                num_above, num_above + self.num_ringassems_master_nodes
            )
        )
        return ringassesm_master_nodes

    @property
    def num_ringassems_master_nodes(self):
        return len(self.ring_assems)

    @property
    def node_ringassesm_idx_dic(self):
        node_ringassesm_idx_dic = {}
        for i, nodes in enumerate(self.ring_assems):
            ring_dic = dict.fromkeys(nodes, i)
            node_ringassesm_idx_dic.update(ring_dic)
        return node_ringassesm_idx_dic

    @property
    def ringassems_master_edges(self):
        ring_belong_to_ring_assems_idx = [
            self.node_ringassesm_idx_dic[ring[0]] for ring in self.new_sssr
        ]
        master_nodes = np.array(self.ringassems_master_nodes, dtype=np.int)
        master_nodes = master_nodes[ring_belong_to_ring_assems_idx]
        ringassems_master_edges = np.stack(
            [master_nodes, self.rings_master_nodes], axis=0
        ).T
        return ringassems_master_edges

    @property
    def mol_master_nodes(self):
        num_above = (
            self.num_atoms +
            self.num_original_bonds +
            self.num_remote_connection_2 +
            self.num_remote_connection_3 +
            self.num_chains_master_nodes +
            self.num_original_rings +
            self.num_ringassems_master_nodes
        )
        return [num_above]

    @property
    def mol_master_edges(self):
        sub_level_nodes = (
            self.chains_master_nodes + self.ringassems_master_nodes
        )
        master_nodes = np.repeat(
            self.mol_master_nodes[0],
            len(sub_level_nodes)
        )
        mol_master_edges = np.stack([master_nodes, sub_level_nodes], axis=0).T
        return mol_master_edges

    @property
    def all_nodes_o(self):
        all_nodes = np.concatenate(
            [
                self.atom_types,

                self.original_bond_types + self.num_atom_types,

                np.zeros(self.num_remote_connection_2, dtype=np.int) +
                self.num_atom_types + self.num_bond_types,

                np.zeros(self.num_remote_connection_3, dtype=np.int) +
                self.num_atom_types + self.num_bond_types + 1,

                np.zeros(self.num_chains_master_nodes, dtype=np.int) +
                self.num_atom_types + self.num_bond_types + 2,

                np.zeros(self.num_rings_master_nodes, dtype=np.int) +
                self.num_atom_types + self.num_bond_types + 3,

                np.zeros(self.num_ringassems_master_nodes, dtype=np.int) +
                self.num_atom_types + self.num_bond_types + 4,

                np.zeros(1, dtype=np.int) +
                self.num_atom_types + self.num_bond_types + 5
            ], axis=-1
        )
        return all_nodes

    @property
    def all_nodes_c(self):
        all_nodes = np.concatenate(
            [
                self.atom_type_c,

                self.original_bond_types_c + 1,

                np.zeros(self.num_remote_connection_2, dtype=np.int) + 2,

                np.zeros(self.num_remote_connection_3, dtype=np.int) + 3,

                np.zeros(self.num_chains_master_nodes, dtype=np.int) + 4,

                np.zeros(self.num_rings_master_nodes, dtype=np.int) + 5,

                np.zeros(self.num_ringassems_master_nodes, dtype=np.int) + 6,

                np.zeros(1, dtype=np.int) + 7
            ], axis=-1
        )
        return all_nodes

    @property
    def all_edges(self):
        all_edges_order = np.concatenate(
            [
                self.new_edge_info,
                self.remote_bonds_2,
                self.remote_bonds_3,
                self.rings_master_edges,
                self.chains_master_edges,
                self.ringassems_master_edges,
                self.mol_master_edges
            ], axis=0
        )
        all_edges_flipped = np.flip(all_edges_order, axis=-1)
        all_edges = np.concatenate(
            [all_edges_order, all_edges_flipped],
            axis=0
        )
        return all_edges

    @property
    def num_all_nodes(self):
        return len(self.all_nodes_o)
