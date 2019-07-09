import typing as t
from copy import deepcopy

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

    @property
    def num_atoms(self):
        return self.mol.GetNumAtoms()

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
        indices = self.original_bond_info.repeat(2, 0)  # 2N_bonds x 2
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
    def sssr(self) -> t.List[t.List]:
        """Get sssr atom indices of a molecule
        
        Returns:
            t.List[t.List]: [[sssr1 atomindices],[sssr2 atom indices], ...]
        """
        return [list(ring) for ring in rdmolops.GetSymmSSSR(self.mol)]

    @property
    def new_rings(self):
        broken_original_bonds = []
        new_bonds = []

        for ring in self.sssr:

        pass

    @property
    def chains(self):
        ls_atom_idx = list(range(self.mol.GetNumAtoms()))
        pass

