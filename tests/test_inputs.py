import pytest


def require_torch():
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("torch not installed", allow_module_level=False)
    return torch


def test_input_generator_shapes():
    torch = require_torch()
    from layer_analysis.inputs.generators import InputGenerator

    gen = InputGenerator(batch_size=2, device=torch.device("cpu"), dtype=torch.float32, seed=42)
    normal = gen.make_normal(2, 3, 4)
    uniform = gen.make_uniform(2, 3, 4)
    assert normal.shape == (2, 3, 4)
    assert uniform.shape == (2, 3, 4)


def test_generate_one_hot_distribution():
    torch = require_torch()
    from layer_analysis.inputs.generators import InputGenerator

    gen = InputGenerator(batch_size=2, device=torch.device("cpu"), dtype=torch.float32, seed=0)
    x = gen.generate("one_hot", (2, 3, 4))
    assert x.shape == (2, 3, 4)
    assert x.sum().item() == pytest.approx(2 * 3, rel=0, abs=0)
