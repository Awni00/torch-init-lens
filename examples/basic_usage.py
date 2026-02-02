"""Basic programmatic usage example for layer_analysis."""

from layer_analysis import AnalysisConfig, AnalysisRunner


def main() -> None:
    config = AnalysisConfig(input_shapes={"x": (2, 16)}, device="cpu")

    runner = AnalysisRunner(
        "torch.nn@Linear",
        config,
        module_kwargs={"in_features": 16, "out_features": 32},
    )

    results = runner.run_all()
    runner.print_summary(results)


if __name__ == "__main__":
    main()
