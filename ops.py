import typing as t

import torch
from torch import nn
from torch_sparse import spspmm

__all__ = [
    'loss_func',
    'spmmsp'
]


def loss_func(recon_x, x, mu1, logvar1, mu2, logvar2):
    loss_recon = nn.MSELoss()
    loss_recon = loss_recon.to(recon_x.device)
    MSE = loss_recon(recon_x, x)

    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5)

    KLD_element = (
        (logvar2 - logvar1).mul(1 / 2).add(
            (logvar1.exp().add((mu1 - mu2).pow(2))).div(2 * logvar2.exp()) -
            1 / 2
        )
    )
    KLD = KLD_element.mean()

    # KL divergence
    return MSE, KLD


def spmmsp(
    sp1: torch.Tensor,
    sp2: torch.Tensor
) -> torch.Tensor:
    assert sp1.size(-1) == sp2.size(0) and sp1.is_sparse and sp2.is_sparse
    m = sp1.size(0)
    k = sp2.size(0)
    n = sp2.size(-1)
    indices, values = spspmm(
        sp1.indices(), sp1.values(),
        sp2.indices(), sp2.values(),
        m, k, n
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        torch.Size([m, n])
    )


def get_activation(
    name: str,
    *args,
    **kwargs
) -> t.Callable:
    """ Get activation module by name

    Args:
        name (str): The name of the activation function (relu, elu, selu)
        args, kwargs: Other parameters

    Returns:
        nn.Module: The activation module
    """
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(*args, **kwargs)
    elif name == 'elu':
        return nn.ELU(*args, **kwargs)
    elif name == 'selu':
        return nn.SELU(*args, **kwargs)
    else:
        raise ValueError('Activation not implemented')
