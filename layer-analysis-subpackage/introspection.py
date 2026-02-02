"""
Module Introspection Utilities.

This module provides utilities for loading, inspecting, and analyzing
PyTorch module classes and instances.
"""

from typing import Type, Dict, Any, List, Tuple, Optional
from pathlib import Path
import importlib
import inspect
import yaml
import torch
import torch.nn as nn


def load_module_class(
    identifier: str,
    prefix: str = "",
) -> Type[nn.Module]:
    """Load a module class from an identifier string.

    Parameters
    ----------
    identifier : str
        Module identifier in format "module.path@ClassName".
        Examples:
        - "models.abstractor@RelationalAttention"
        - "torch.nn@Linear"
    prefix : str
        Optional prefix to prepend to module path.

    Returns
    -------
    Type[nn.Module]
        The loaded module class.

    Raises
    ------
    ValueError
        If identifier format is invalid.
    ModuleNotFoundError
        If the module path cannot be imported.
    AttributeError
        If the class name is not found in the module.
    TypeError
        If the class is not an nn.Module subclass.

    Examples
    --------
    >>> cls = load_module_class("models.abstractor@RelationalAttention")
    >>> module = cls(d_model=64, n_heads=4)
    """
    from .errors import ModuleLoadError

    if "@" not in identifier:
        raise ModuleLoadError(
            identifier,
            "Identifier must be in format 'module.path@ClassName'",
        )

    module_path, class_name = identifier.split("@", 1)

    full_path = prefix + module_path if prefix else module_path

    try:
        module = importlib.import_module(full_path)
    except ModuleNotFoundError as e:
        raise ModuleLoadError(identifier, f"Module not found: {e}")

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ModuleLoadError(
            identifier,
            f"Class '{class_name}' not found in module '{full_path}'",
        )

    if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
        raise ModuleLoadError(
            identifier,
            f"'{class_name}' is not an nn.Module subclass",
        )

    return cls


def get_module_signature(cls: Type[nn.Module]) -> Dict[str, Any]:
    """Extract constructor signature and defaults.

    Parameters
    ----------
    cls : Type[nn.Module]
        Module class to inspect.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping parameter names to their metadata:
        - "default": Default value or None if required
        - "annotation": Type annotation as string or None
        - "required": Whether the parameter is required

    Examples
    --------
    >>> from torch.nn import Linear
    >>> sig = get_module_signature(Linear)
    >>> sig["in_features"]
    {'default': None, 'annotation': 'int', 'required': True}
    """
    sig = inspect.signature(cls.__init__)
    params = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        default = (
            param.default
            if param.default is not inspect.Parameter.empty
            else None
        )
        annotation = (
            str(param.annotation)
            if param.annotation is not inspect.Parameter.empty
            else None
        )
        required = param.default is inspect.Parameter.empty

        params[name] = {
            "default": default,
            "annotation": annotation,
            "required": required,
        }

    return params


def get_required_params(cls: Type[nn.Module]) -> List[str]:
    """Get list of required constructor parameters.

    Parameters
    ----------
    cls : Type[nn.Module]
        Module class to inspect.

    Returns
    -------
    List[str]
        Names of required parameters (excluding 'self').
    """
    sig = get_module_signature(cls)
    return [name for name, info in sig.items() if info["required"]]


def load_defaults() -> Dict[str, Any]:
    """Load default hyperparameters from defaults.yaml.

    Returns
    -------
    Dict[str, Any]
        Dictionary of default parameter values.
    """
    defaults_path = Path(__file__).parent / "defaults.yaml"
    if not defaults_path.exists():
        return {}
    with open(defaults_path) as f:
        return yaml.safe_load(f) or {}


def merge_kwargs(
    module_class: Type[nn.Module],
    user_kwargs: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Merge hyperparameters with priority: user kwargs > derived > defaults.

    Only includes parameters that exist in the module's __init__ signature.

    Parameters
    ----------
    module_class : Type[nn.Module]
        Module class to instantiate.
    user_kwargs : Dict[str, Any]
        User-provided keyword arguments.
    input_shape : Optional[Tuple[int, ...]]
        Input tensor shape, used to derive d_model and hidden_size.

    Returns
    -------
    Dict[str, Any]
        Merged kwargs ready for module instantiation.
    """
    sig = get_module_signature(module_class)
    defaults = load_defaults()

    kwargs = {}

    # 1. Apply applicable YAML defaults (lowest priority)
    for param_name, default_value in defaults.items():
        if param_name in sig:
            kwargs[param_name] = default_value

    # 2. Apply derived values from input_shape
    if input_shape is not None and len(input_shape) > 2:
        d_model = input_shape[2]
        if "d_model" in sig:
            kwargs["d_model"] = d_model
        if "hidden_size" in sig:
            kwargs["hidden_size"] = d_model

    # 3. Override with user-provided kwargs (highest priority)
    kwargs.update(user_kwargs)

    return kwargs


def validate_kwargs(
    module_class: Type[nn.Module],
    kwargs: Dict[str, Any],
    user_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate that all required parameters are provided and no invalid ones.

    Parameters
    ----------
    module_class : Type[nn.Module]
        Module class to validate against.
    kwargs : Dict[str, Any]
        Merged keyword arguments to validate (for checking required params).
    user_kwargs : Optional[Dict[str, Any]]
        Original user-provided kwargs (for checking invalid params).
        If None, invalid kwarg checking is skipped.

    Raises
    ------
    MissingModuleKwargsError
        If required parameters are missing.
    InvalidModuleKwargsError
        If invalid parameters are provided.
    """
    from .errors import MissingModuleKwargsError, InvalidModuleKwargsError

    sig = get_module_signature(module_class)

    # Check for invalid kwargs (only from user-provided kwargs)
    if user_kwargs is not None:
        invalid = [p for p in user_kwargs if p not in sig]
        if invalid:
            raise InvalidModuleKwargsError(
                module_class.__name__,
                invalid,
                sig,
            )

    # Check for missing required kwargs
    required = get_required_params(module_class)
    missing = [p for p in required if p not in kwargs]

    if missing:
        raise MissingModuleKwargsError(
            module_class.__name__,
            missing,
            sig,
        )


def detect_module_type(module: nn.Module) -> str:
    """Detect the type of module for appropriate analysis.

    Parameters
    ----------
    module : nn.Module
        Module instance to classify.

    Returns
    -------
    str
        Module type classification:
        - "attention": Attention mechanisms
        - "linear": Linear/Dense layers
        - "embedding": Embedding layers
        - "normalization": Normalization layers (LayerNorm, RMSNorm, etc.)
        - "activation": Activation functions
        - "convolution": Convolutional layers
        - "recurrent": RNN/LSTM/GRU layers
        - "composite": Modules with multiple submodules
        - "unknown": Unclassified modules

    Examples
    --------
    >>> detect_module_type(nn.Linear(10, 5))
    'linear'
    >>> detect_module_type(nn.MultiheadAttention(64, 4))
    'attention'
    """
    class_name = type(module).__name__.lower()
    base_classes = [c.__name__.lower() for c in type(module).__mro__]

    # Check by class name patterns
    if any(x in class_name for x in ["attention", "attn", "mha"]):
        return "attention"

    if any(x in class_name for x in ["linear", "dense", "projection"]):
        return "linear"

    if any(x in class_name for x in ["embedding", "embed"]):
        return "embedding"

    if any(
        x in class_name
        for x in ["norm", "layernorm", "rmsnorm", "batchnorm", "groupnorm"]
    ):
        return "normalization"

    if any(
        x in class_name
        for x in ["relu", "gelu", "silu", "sigmoid", "tanh", "activation"]
    ):
        return "activation"

    if any(x in class_name for x in ["conv1d", "conv2d", "conv3d", "conv"]):
        return "convolution"

    if any(x in class_name for x in ["rnn", "lstm", "gru", "recurrent"]):
        return "recurrent"

    # Check if it's a container/composite module
    submodules = list(module.children())
    if len(submodules) > 2:
        return "composite"

    return "unknown"


def enumerate_layers(
    module: nn.Module,
    include_containers: bool = False,
) -> List[Tuple[str, nn.Module]]:
    """Enumerate all named layers in a module.

    Parameters
    ----------
    module : nn.Module
        Module to enumerate.
    include_containers : bool
        Whether to include container modules (Sequential, ModuleList, etc.)

    Returns
    -------
    List[Tuple[str, nn.Module]]
        List of (name, submodule) pairs.

    Examples
    --------
    >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
    >>> layers = enumerate_layers(model)
    >>> [(name, type(m).__name__) for name, m in layers]
    [('0', 'Linear'), ('1', 'ReLU')]
    """
    container_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
    layers = []

    for name, submodule in module.named_modules():
        if not name:  # Skip the root module
            continue
        if not include_containers and isinstance(submodule, container_types):
            continue
        layers.append((name, submodule))

    return layers


def get_input_signature(module: nn.Module) -> Optional[Dict[str, Any]]:
    """Try to infer the forward method's input signature.

    Parameters
    ----------
    module : nn.Module
        Module to inspect.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary mapping input parameter names to their metadata,
        or None if signature cannot be determined.
    """
    forward_method = getattr(module, "forward", None)
    if forward_method is None:
        return None

    try:
        sig = inspect.signature(forward_method)
    except (ValueError, TypeError):
        return None

    inputs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        annotation = (
            str(param.annotation)
            if param.annotation is not inspect.Parameter.empty
            else None
        )
        default = (
            param.default
            if param.default is not inspect.Parameter.empty
            else None
        )

        inputs[name] = {
            "annotation": annotation,
            "default": default,
            "kind": str(param.kind),
            "required": param.default is inspect.Parameter.empty,
        }

    return inputs


def count_parameters(
    module: nn.Module,
    trainable_only: bool = False,
) -> int:
    """Count total number of parameters in a module.

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    trainable_only : bool
        If True, only count parameters with requires_grad=True.

    Returns
    -------
    int
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def get_parameter_info(module: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Get detailed information about all parameters.

    Parameters
    ----------
    module : nn.Module
        Module to analyze.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping parameter names to their metadata:
        - "shape": Parameter shape as tuple
        - "numel": Number of elements
        - "dtype": Data type
        - "requires_grad": Whether parameter is trainable
        - "device": Device location
    """
    info = {}
    for name, param in module.named_parameters():
        info[name] = {
            "shape": tuple(param.shape),
            "numel": param.numel(),
            "dtype": str(param.dtype),
            "requires_grad": param.requires_grad,
            "device": str(param.device),
        }
    return info


def get_buffer_info(module: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Get detailed information about all buffers.

    Parameters
    ----------
    module : nn.Module
        Module to analyze.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping buffer names to their metadata.
    """
    info = {}
    for name, buffer in module.named_buffers():
        info[name] = {
            "shape": tuple(buffer.shape),
            "numel": buffer.numel(),
            "dtype": str(buffer.dtype),
            "device": str(buffer.device),
        }
    return info


def infer_device(module: nn.Module) -> torch.device:
    """Infer device from module parameters or buffers.

    Parameters
    ----------
    module : nn.Module
        Module to inspect.

    Returns
    -------
    torch.device
        Device of the first parameter/buffer, or CPU if none found.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass

    try:
        return next(module.buffers()).device
    except StopIteration:
        pass

    return torch.device("cpu")


def infer_dtype(module: nn.Module) -> torch.dtype:
    """Infer dtype from module parameters.

    Parameters
    ----------
    module : nn.Module
        Module to inspect.

    Returns
    -------
    torch.dtype
        Dtype of the first parameter, or float32 if none found.
    """
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32
