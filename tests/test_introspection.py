import pytest


def require_torch():
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("torch not installed", allow_module_level=False)
    return torch


def test_load_module_class_linear():
    torch = require_torch()
    from layer_analysis.introspection import load_module_class

    cls = load_module_class("torch.nn@Linear")
    assert issubclass(cls, torch.nn.Module)


def test_get_module_signature_includes_required_fields():
    torch = require_torch()
    from layer_analysis.introspection import get_module_signature

    sig = get_module_signature(torch.nn.Linear)
    assert "in_features" in sig
    assert sig["in_features"]["required"] is True


def test_validate_kwargs_missing_required():
    require_torch()
    from layer_analysis.errors import MissingModuleKwargsError
    from layer_analysis.introspection import load_module_class, validate_kwargs

    cls = load_module_class("torch.nn@Linear")
    with pytest.raises(MissingModuleKwargsError):
        validate_kwargs(cls, kwargs={}, user_kwargs={})
