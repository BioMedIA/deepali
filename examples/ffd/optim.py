import torch.optim
from torch.optim import Optimizer
from torch.nn import Module


def new_optimizer(name: str, model: Module, **kwargs) -> Optimizer:
    r"""Initialize new optimizer for parameters of given model.

    Args:
        name: Name of optimizer.
        model: Module whose parameters are to be optimized.
        kwargs: Keyword arguments for named optimizer.

    Returns:
        New optimizer instance.

    """
    cls = getattr(torch.optim, name, None)
    if cls is None:
        raise ValueError(f"Unknown optimizer: {name}")
    if not issubclass(cls, Optimizer):
        raise TypeError(f"Requested type '{name}' is not a subclass of torch.optim.Optimizer")
    if "learning_rate" in kwargs:
        if "lr" in kwargs:
            raise ValueError("new_optimizer() 'lr' and 'learning_rate' are mutually exclusive")
        kwargs["lr"] = kwargs.pop("learning_rate")
    return cls(model.parameters(), **kwargs)
