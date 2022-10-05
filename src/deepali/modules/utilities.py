r"""Auxiliary functions for torch.nn.Module instances."""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from torch.nn import Module


def has_children(module: Module) -> bool:
    r"""Check if module has other modules as children."""
    try:
        next(iter(module.children()))
    except StopIteration:
        return False
    return True


def module_ids_by_class_name(
    model: Module,
    duplicates: bool = False,
    containers: bool = False,
    memo: Optional[dict] = None,
) -> Dict[str, List[int]]:
    r"""Obtain ids of module objects in model, indexed by module type name.

    Args:
        model: Neural network model.
        duplicates: Whether to return lists of object ids instead of sets.
        containers: Whether to include object ids of modules with children.
        memo: Used internally to recursively collect object ids.

    Returns:
        Dictionary with module class names as keys and object ids of each type stored list.

    """
    if memo is None:
        memo = defaultdict(list)
    modules: dict = model._modules
    for module in modules.values():
        assert isinstance(module, Module)
        if containers or not has_children(module):
            ids = memo[module.__class__.__name__]
            ids.append(id(module))
        module_ids_by_class_name(module, duplicates=duplicates, containers=containers, memo=memo)
    if not duplicates:
        memo = {k: list(set(v)) for k, v in memo.items()}
    return memo


def module_counts_by_class_name(
    model: Module, duplicates: bool = False, containers: bool = False
) -> Dict[str, int]:
    r"""Count how many module objects are in a model of each module type.

    Args:
        model: Neural network model.
        duplicates: Whether to count duplicate objects.
        containers: Whether to include modules with children.

    Returns:
        Dictionary with module class names as keys and counts of object ids as values.

    """
    ids = module_ids_by_class_name(model, duplicates=duplicates, containers=containers)
    return {k: len(v) for k, v in ids.items()}


def rename_layers_in_state_dict(state: Dict[str, Any], rename: Dict[str, str]) -> Dict[str, Any]:
    r"""Rename layers in loaded state dict."""
    metadata = getattr(state, "_metadata", None)
    state = state.copy()  # does not copy _metadata attribute
    if metadata is not None:
        state._metadata = rename_layers_in_state_dict(metadata, rename)
    for prefix, new_name in rename.items():
        for key in list(state.keys()):
            if key == prefix or key.startswith(prefix + "."):
                new_key = new_name + key[len(prefix) :]
                state[new_key] = state.pop(key)
    return state
