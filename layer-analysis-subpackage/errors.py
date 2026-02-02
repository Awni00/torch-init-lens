"""
Custom Exceptions for Module Initialization Analysis.

This module defines exception classes for various error conditions that
can occur during module analysis.
"""

from typing import Optional, Tuple
import torch


class ModuleAnalysisError(Exception):
    """Base exception for module analysis errors."""

    pass


class ModuleLoadError(ModuleAnalysisError):
    """Failed to load module class.

    Raised when the module identifier cannot be resolved to a valid
    nn.Module subclass.
    """

    def __init__(self, identifier: str, reason: str = ""):
        self.identifier = identifier
        self.reason = reason
        message = f"Failed to load module '{identifier}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class IncompatibleModuleError(ModuleAnalysisError):
    """Module is not compatible with requested analysis.

    Raised when the module structure is incompatible with a specific
    analysis type (e.g., no 2D weights for spectral analysis).
    """

    def __init__(self, analysis_type: str, reason: str = ""):
        self.analysis_type = analysis_type
        self.reason = reason
        message = f"Module incompatible with {analysis_type} analysis"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class NumericalInstabilityError(ModuleAnalysisError):
    """Numerical instability detected during analysis.

    Raised when computations fail due to numerical issues such as
    singular matrices, overflow, or NaN values.
    """

    def __init__(
        self,
        operation: str,
        details: str = "",
        values: Optional[torch.Tensor] = None,
    ):
        self.operation = operation
        self.details = details
        self.values = values
        message = f"Numerical instability in {operation}"
        if details:
            message += f": {details}"
        super().__init__(message)


class GradientError(ModuleAnalysisError):
    """Gradient computation failed.

    Raised when gradient computation fails or produces unexpected results.
    """

    def __init__(
        self,
        param_name: Optional[str] = None,
        reason: str = "",
    ):
        self.param_name = param_name
        self.reason = reason
        message = "Gradient computation failed"
        if param_name:
            message += f" for parameter '{param_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class InputShapeError(ModuleAnalysisError):
    """Input shape is incompatible with module.

    Raised when the configured input shape cannot be used with the module.
    """

    def __init__(
        self,
        expected_shape: Tuple[int, ...],
        actual_shape: Tuple[int, ...],
        reason: str = "",
    ):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.reason = reason
        message = f"Input shape mismatch: expected {expected_shape}, got {actual_shape}"
        if reason:
            message += f". {reason}"
        super().__init__(message)


class MissingModuleKwargsError(ModuleAnalysisError):
    """Required module kwargs are missing.

    Raised when module instantiation requires kwargs that were not provided.
    """

    def __init__(
        self,
        module_identifier: str,
        missing_kwargs: list,
        signature_info: Optional[dict] = None,
    ):
        self.module_identifier = module_identifier
        self.missing_kwargs = missing_kwargs
        self.signature_info = signature_info

        message = (
            f"Missing required module kwargs for '{module_identifier}': "
            f"{', '.join(missing_kwargs)}"
        )
        if signature_info:
            sig_lines = []
            for name, info in signature_info.items():
                req = "(required)" if info.get("required") else f"(default={info.get('default')!r})"
                sig_lines.append(f"  - {name} {req}")
            message += f"\n\nConstructor signature:\n" + "\n".join(sig_lines)

        super().__init__(message)


class InvalidModuleKwargsError(ModuleAnalysisError):
    """Invalid module kwargs provided.

    Raised when kwargs are provided that do not exist in the module's signature.
    """

    def __init__(
        self,
        module_identifier: str,
        invalid_kwargs: list,
        signature_info: Optional[dict] = None,
    ):
        self.module_identifier = module_identifier
        self.invalid_kwargs = invalid_kwargs
        self.signature_info = signature_info

        message = (
            f"Invalid module kwargs for '{module_identifier}': "
            f"{', '.join(invalid_kwargs)}"
        )
        if signature_info:
            sig_lines = []
            for name, info in signature_info.items():
                req = "(required)" if info.get("required") else f"(default={info.get('default')!r})"
                sig_lines.append(f"  - {name} {req}")
            message += f"\n\nValid parameters:\n" + "\n".join(sig_lines)

        super().__init__(message)


class AnalysisTimeoutError(ModuleAnalysisError):
    """Analysis took too long to complete.

    Raised when an analysis operation exceeds the configured timeout.
    """

    def __init__(self, analysis_type: str, timeout_seconds: float):
        self.analysis_type = analysis_type
        self.timeout_seconds = timeout_seconds
        message = f"{analysis_type} analysis timed out after {timeout_seconds}s"
        super().__init__(message)


def safe_svd(
    weight: torch.Tensor,
    max_attempts: int = 3,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Compute SVD with fallback for numerical issues.

    Attempts SVD computation with increasingly robust methods:
    1. Standard SVD in original dtype
    2. SVD in float64 precision
    3. Power iteration for largest singular value only

    Parameters
    ----------
    weight : torch.Tensor
        Weight matrix to decompose.
    max_attempts : int
        Number of retry attempts with different methods.

    Returns
    -------
    Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]
        (U, S, Vh) singular value decomposition. U and Vh may be None
        if only singular values were computed via power iteration.

    Raises
    ------
    NumericalInstabilityError
        If SVD fails after all attempts.
    """
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # Try standard SVD
                return torch.linalg.svd(weight, full_matrices=False)
            elif attempt == 1:
                # Try with float64 precision
                weight_f64 = weight.double()
                U, S, Vh = torch.linalg.svd(weight_f64, full_matrices=False)
                return (
                    U.to(weight.dtype),
                    S.to(weight.dtype),
                    Vh.to(weight.dtype),
                )
            else:
                # Try randomized SVD / power iteration for large matrices
                if weight.numel() > 10000:
                    # Simple power iteration for largest singular value
                    v = torch.randn(
                        weight.size(1),
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    for _ in range(20):
                        u = weight @ v
                        u = u / (torch.linalg.norm(u) + 1e-8)
                        v = weight.T @ u
                        v = v / (torch.linalg.norm(v) + 1e-8)
                    sigma = torch.linalg.norm(weight @ v)
                    return None, sigma.unsqueeze(0), None
                else:
                    # Small matrix, retry with regularization
                    weight_reg = weight + 1e-6 * torch.eye(
                        min(weight.shape),
                        device=weight.device,
                        dtype=weight.dtype,
                    )[: weight.size(0), : weight.size(1)]
                    return torch.linalg.svd(weight_reg, full_matrices=False)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise NumericalInstabilityError(
                    "SVD",
                    f"Failed after {max_attempts} attempts: {e}",
                )

    raise NumericalInstabilityError("SVD", "Failed unexpectedly")


def safe_forward(
    module: torch.nn.Module,
    *args: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Execute forward pass with error handling.

    Supports modules with multiple positional tensor inputs.

    Parameters
    ----------
    module : torch.nn.Module
        Module to execute.
    *args : torch.Tensor
        Input tensor(s). Can be a single tensor or multiple positional tensors.
    **kwargs
        Additional arguments for the forward pass.

    Returns
    -------
    torch.Tensor
        Output tensor (first element if tuple).

    Raises
    ------
    ModuleAnalysisError
        If forward pass fails.

    Examples
    --------
    >>> # Single input
    >>> out = safe_forward(module, x)
    >>> # Multiple inputs (e.g., RelationalAttention)
    >>> out = safe_forward(module, x, symbols)
    """
    try:
        with torch.no_grad():
            out = module(*args, **kwargs)
            if isinstance(out, tuple):
                # Return first non-None tensor
                for o in out:
                    if isinstance(o, torch.Tensor):
                        return o
                return out[0]
            return out
    except Exception as e:
        raise ModuleAnalysisError(f"Forward pass failed: {e}")


def check_for_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor",
) -> None:
    """Check tensor for NaN or Inf values.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check.
    name : str
        Name of the tensor for error messages.

    Raises
    ------
    NumericalInstabilityError
        If tensor contains NaN or Inf values.
    """
    if torch.isnan(tensor).any():
        raise NumericalInstabilityError(
            f"NaN check for {name}",
            f"Found {torch.isnan(tensor).sum().item()} NaN values",
            tensor,
        )
    if torch.isinf(tensor).any():
        raise NumericalInstabilityError(
            f"Inf check for {name}",
            f"Found {torch.isinf(tensor).sum().item()} Inf values",
            tensor,
        )
