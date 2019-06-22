import linecache

import dgl
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from mol_spec import *
from ops import *

__all__ = [
    'smiles_to_dgl_graph',
    'ms',
    'graph_from_line',
    'get_num_lines',
    'str_from_line',
    'graph_to_whole_graph',   
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


def get_remote_connection(
    adj: torch.Tensor,
) -> t.Tuple[torch.Tensor, ...]:
    d = spmmsp(adj.coalesce(), adj.coalesce())
    d_indices_2 = d.to_dense().nonzero().t()
    d_indices_2 = d_indices_2[:, d_indices_2[0, :] != d_indices_2[1, :]]
    
    d = spmmsp(d.coalesce(), adj.coalesce())
    d = d - d.mul(adj)
    d_indices_3 = d.to_dense().nonzero().t()
    d_indices_3 = d_indices_3[:, d_indices_3[0, :] != d_indices_3[1, :]]
    return d_indices_2, d_indices_3


def graph_to_whole_graph(
    adj: torch.Tensor,
    bond_info: torch.Tensor,
    n_feat: torch.Tensor,
    e_feat: torch.Tensor
) -> t.Tuple[torch.Tensor, ...]:
    """Involving remote connections and consider edges as nodes 
    
    Args:
        adj (torch.Tensor): 
            adj with out self connections N x N
        bond_info (torch.Tensor): 
            original bond info 2 x N_e
        n_feat (torch.Tensor): 
            original node feat N x F
        e_feat (torch.Tensor): 
            original edge feat N_e x F_e
    """
    # adj = g.adjacency_matrix()
    # bond_info = torch.stack(g.edges(), dim=0)
    # n_feat = g.ndata['feat']
    # e_feat = g.edata['feat']

    num_n_feat, num_e_feat = n_feat.size(-1), e_feat.size(-1)
    d_indices_2, d_indices_3 = get_remote_connection(adj)
    all_bond_info = torch.cat([bond_info, d_indices_2, d_indices_3], dim=-1)
    all_e_feat = torch.cat(
        [
            torch.cat(
                [
                    e_feat, 
                    torch.zeros([e_feat.size(0), 2])
                ], 
                dim=-1
            ),
            torch.cat(
                [
                    torch.zeros([d_indices_2.size(-1), e_feat.size(-1)]),
                    torch.ones([d_indices_2.size(-1), 1]),
                    torch.zeros([d_indices_2.size(-1), 1])
                ], 
                dim=-1
            ),
            torch.cat(
                [
                    torch.zeros([d_indices_3.size(-1), e_feat.size(-1)]),
                    torch.zeros([d_indices_3.size(-1), 1]),
                    torch.ones([d_indices_3.size(-1), 1])
                ], 
                dim=-1
            )
        ],
        dim=0
    )
    num_n = n_feat.size(0)
    num_e = all_e_feat.size(0)
    ndata_new = torch.cat(
        (n_feat, torch.zeros(num_n, all_e_feat.size(1))),
        dim=1
    )
    edata_new = torch.cat(
        (torch.zeros([all_e_feat.size(0), num_n_feat]), all_e_feat),
        dim=1
    )
    all_node_data = torch.cat(
        (ndata_new, edata_new),
        dim=0
    )
    n_new = torch.arange(
        num_n, 
        all_node_data.size(0)
    )
    all_new_bond_info = torch.cat(
        [
            torch.stack(
                [all_bond_info[0], n_new],
                dim=0
            ),
            torch.stack(
                [n_new, all_bond_info[0]],
                dim=0
            ),
            torch.stack(
                [all_bond_info[1], n_new],
                dim=0
            ),
            torch.stack(
                [n_new, all_bond_info[1]],
                dim=0
            )
        ],
        dim=-1
    )
    adj = (
        torch.eye(all_node_data.size(0)).to_sparse() +
        torch.sparse_coo_tensor(
            all_new_bond_info,
            [1. for _ in range(all_new_bond_info.size(-1))],
            torch.Size(
                [all_node_data.size(0), all_node_data.size(0)]
            )
        )
    )
    return all_node_data, all_new_bond_info, adj


# def get_whole_data(
#     g: dgl.graph.DGLGraph,
#     key_n: str='feat',
#     key_e: str='feat'
# ):
#     num_n_feat, num_e_feat = g.ndata[key_n].size(-1), g.edata[key_e].size(-1)
#     # num_feat = num_n_feat + num_e_feat
#     num_n, num_e = g.ndata[key_n].size(0), int(g.edata[key_e].size(0) / 2)
#     batch_size = num_n + num_e
#     ndata_new = torch.cat(
#         (g.ndata[key_n], torch.zeros(num_n, num_e_feat)),
#         dim=1
#     )
#     edata_new = torch.cat(
#         (torch.zeros([num_e, num_n_feat]), g.edata[key_e][: num_e]),
#         dim=1
#     )
#     all_node_data = torch.cat(
#         (ndata_new, edata_new),
#         dim=0
#     )

#     n_new = torch.arange(num_n, batch_size)
#     indices = torch.cat(
#         [n_new for _ in range(2)],
#         dim=-1
#     )
#     indices1 = torch.stack(
#         [indices, g.edges()[0]],
#         dim=0
#     )
#     indices2 = torch.stack(
#         [g.edges()[0], indices],
#         dim=0
#     )
#     indices = torch.cat([indices1, indices2], dim=-1)
#     n_e_adj = torch.sparse_coo_tensor(
#         indices,
#         [1. for _ in range(indices.size(-1))],
#         torch.Size([batch_size, batch_size])
#     )
#     adj = (
#         torch.eye(batch_size).to_sparse() +
#         n_e_adj
#     )

#     return all_node_data, adj
