import typing as t

from torch import nn


def get_activation(
    name: str,
    *args,
    **kwargs
):
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
