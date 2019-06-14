import linecache

import dgl
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from mol_spec import *

__all__ = [
    'smiles_to_dgl_graph',
    'ms',
    'graph_from_line',
]

ms = MoleculeSpec.get_default()


def smiles_to_dgl_graph(
    smiles: str, 
    ms: MoleculeSpec = ms,
    ranked: bool = False
) -> dgl.DGLGraph:
    """Convert smiles to dgl graph
    
    Args:
        smiles (str): a molecule smiles
        ms (MoleculeSpec, optional): MoleculeSpec
    """
    # smiles = standardize_smiles(smiles)
    m = AllChem.MolFromSmiles(smiles)

    m = Chem.RemoveHs(m)
    g = dgl.DGLGraph()
    g.add_nodes(m.GetNumAtoms())

    ls_edge = [
        (e.GetBeginAtomIdx(), e.GetEndAtomIdx()) for e in m.GetBonds()
    ]
    if ranked:
        ls_atom_type = [
            ms.atom_types.index(('C', 0, 0)) for _ in m.GetAtoms()
        ]
        ls_edge_type = [
            ms.bond_orders.index(rdkit.Chem.rdchem.BondType.SINGLE)
            for _ in m.GetBonds()
        ]
    else:
        ls_atom_type = [
            ms.get_atom_type(atom) for atom in m.GetAtoms()
        ]
        ls_edge_type = [
            ms.get_bond_type(bond) for bond in m.GetBonds()
        ]
    src, dst = tuple(zip(*ls_edge))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.ndata['feat'] = label_to_onehot(
        ls_atom_type, 
        len(ms.atom_types)
    )
    g.edata['feat'] = label_to_onehot(
        ls_edge_type, 
        len(ms.bond_orders)
    ).repeat(2, 1)
    return g


def label_to_onehot(ls, class_num):
    ls = torch.LongTensor(ls).reshape(-1, 1)
    return torch.zeros(len(ls), class_num).scatter_(1, ls, 1)


def onehot_to_label(tensor):
    return torch.argmax(tensor, dim=1)


def str_from_line(
    file: str, 
    idx: int
) -> str:
    """
    Get string from a specific line
    Args:
        idx (int): index of line
        file (string): location of a file

    Returns:

    """
    return linecache.getline(file, idx + 1).strip()


def get_num_lines(
    input_file: str
):
    """Get num_of_lines of a text file
    Args:
        input_file (str): location of the file

    Returns: num_lines of the file

    Examples:
        >>> get_num_lines("./dataset.txt")
    """
    for num_lines, line in enumerate(open(input_file, 'r')):
        pass
    return num_lines + 1


def graph_from_line(
    file: str,
    idx: int,
    ranked: bool = False
) -> dgl.DGLGraph:
    smiles = str_from_line(
        file,
        idx
    )
    g = smiles_to_dgl_graph(
        smiles,
        ms=ms,
        ranked=ranked
    )
    return g


