import pytest


def require_torch():
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("torch not installed", allow_module_level=False)
    return torch


def test_runner_parameter_norms_only():
    require_torch()
    from layer_analysis import AnalysisConfig, AnalysisRunner

    config = AnalysisConfig(
        input_shapes={"x": (2, 16)},
        device="cpu",
        run_operator_norm=False,
        run_gradient_analysis=False,
        run_spectral_analysis=False,
        run_rank_analysis=False,
        run_lipschitz=False,
        run_activation_stats=False,
        run_gradient_ratios=False,
        run_precision_checks=False,
        run_weight_distribution=False,
    )

    runner = AnalysisRunner(
        "torch.nn@Linear",
        config,
        module_kwargs={"in_features": 16, "out_features": 32},
    )

    results = runner.run_all()
    assert "parameter_norms" in results
    assert len(results["parameter_norms"]) > 0
