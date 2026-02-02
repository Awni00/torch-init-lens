# Layer Analysis CLI Output Documentation

*Version: Commit 6cfdf9853cfaba7813da7064ee01b6c7d0dc0b2e (2026-01-19)*

This document explains the output produced by the `layer_analysis` CLI tool. This tool is designed to verify that PyTorch modules are well-behaved at initialization, helping to catch issues like vanishing gradients, exploding activations, or poor weight distributions before training begins.

## Output Sections

### 1. Parameter Norms
Analyzes the magnitude and scale of the model's parameters.

*   **Frobenius**: The L2 norm of the flattened parameter tensor. Represents the overall "energy" of the weights.
*   **Operator**: The spectral norm (largest singular value) for 2D matrices. For 1D tensors, it matches the Frobenius norm. This measures the maximum amplification factor of the matrix.
*   **Sup**: The supremum norm (maximum absolute value). Useful for checking bounds.
*   **Std**: Standard deviation of the parameter values.
*   **Mean**: Average parameter value. Should typically be close to 0 for initialized weights.

### 2. Operator Norm
Estimates the effective operator norm of the module: $\sup_{x \neq 0} \frac{\|f(x)\|}{\|x\|}$.
This measures how much the module amplifies or diminishes input signals.

*   **Distributions**: tested against various input distributions:
    *   `normal`: Standard Gaussian inputs.
    *   `sparse`: Inputs with many zeros.
    *   `large_magnitude`: Inputs with large values.
    *   `small_magnitude`: Inputs with tiny values.
    *   `correlated`: Inputs with high correlation between features.
*   **Metrics**:
    *   **max**: The maximum amplification observed.
    *   **mean**: The average amplification.

### 3. Gradient Analysis
Ensures that the module is properly connected to the computation graph and that gradients flow correctly.

*   **Gradient Existence**: Checks if `param.grad` is populated for all trainable parameters.
*   **Gradient Norms**: The L2 norm of the gradients.
*   **Gradient Completeness**: Checks for "dead" parameters (zero gradients) or "vanishing" gradients (tiny norms).
*   **Input Gradient**: The norm of the gradient with respect to the input. This indicates how sensitive the loss is to input changes.

### 4. Spectral Analysis
Examines the singular values of 2D weight matrices (e.g., Linear layer weights).

*   **Spectral Radius**: The largest singular value ($\sigma_{\max}$). Determines the maximum gain in any direction.
*   **Condition Number**: The ratio $\sigma_{\max} / \sigma_{\min}$. High values (> 1000) indicate an ill-conditioned matrix, which can lead to numerical instability and slow training.

### 5. Rank Analysis
Checks if weight matrices are full-rank or collapsed.

*   **Effective Rank**: The number of singular values significantly greater than zero (above a tolerance).
*   **Rank Ratio**: Effective Rank / Theoretical Max Rank. Should typically be close to 100% at initialization. Low rank implies the layer is not utilizing its full capacity.

### 6. Lipschitz Estimation
Estimates the Lipschitz constant $L$ such that $\|f(x) - f(y)\| \le L \|x - y\|$.

*   **Empirical**: Estimated by perturbing inputs and measuring output changes ($ \max \frac{\|f(x+\epsilon) - f(x)\|}{\|\epsilon\|} $).
*   **Power Iteration**: Uses an iterative method on the Jacobian to find the largest singular value.
*   **Weight Bound**: A theoretical upper bound calculated as the product of the spectral norms of all layers.

### 7. Activation Statistics
Hooks into the module to analyze intermediate activations during a forward pass.

*   **Mean**: Average activation value.
*   **Std**: Standard deviation of activations.
*   **Max**: Maximum activation value.
*   **Zero Fraction**: Percentage of activations that are zero (e.g., due to ReLU). A very high fraction (> 90%) indicates "dead neurons".

### 8. Gradient Ratios
Analyzes the flow of gradients between layers to detect vanishing or exploding gradients.

*   **Ratio**: The ratio of the gradient norm of a layer to the gradient norm of the preceding layer.
    *   **Vanishing (< 0.1)**: Signal is dying out as it propagates back.
    *   **Exploding (> 10)**: Signal is growing uncontrollably.
    *   **Healthy**: Ratios close to 1.0.

### 9. Precision Checks
Scans for numerical anomalies.

*   **Parameters**: Checks for NaNs, Infs, or extremely large/small values in weights.
*   **Normal/Extreme Inputs**: Checks if outputs contain NaNs/Infs under standard or extreme input conditions.
*   **Gradients**: Checks if gradients contain NaNs/Infs.

### 10. Weight Distribution
Statistical analysis of the weight values.

*   **Mean/Std**: Basic moments.
*   **Skewness**: Measure of asymmetry.
*   **Kurtosis**: Measure of "tailedness".

## Interpreting Results

| Metric | Healthy Range | Warning Signs | Potential Fix |
| :--- | :--- | :--- | :--- |
| **Operator Norm** | 0.5 - 2.0 | >> 1 (Exploding)<br><< 1 (Vanishing) | Adjust init gain or use orthogonal init. |
| **Condition Number** | < 1000 | > 1000 (Ill-conditioned) | Use orthogonal initialization. |
| **Rank Ratio** | > 90% | < 50% (Low Rank) | Check for degenerate init (e.g., all zeros). |
| **Gradient Ratio** | 0.1 - 10 | < 0.1 (Vanishing)<br>> 10 (Exploding) | Add normalization (LayerNorm) or residual connections. |
| **Dead Neurons** | < 50% | > 90% (Dead ReLU) | Lower learning rate or change activation function. |
| **Lipschitz** | ~ 1.0 | >> 1.0 (Unstable) | Spectral normalization or gradient clipping. |

## CLI Examples

```bash
# Basic analysis
python -m layer_analysis analyze torch.nn@Linear \
    --input-shape 2,16 --module-kwargs in_features=16 out_features=32

# Analysis with multiple inputs
python -m layer_analysis analyze torch.nn@MultiheadAttention \
    --input-shapes query:2,8,32 key:2,8,32 value:2,8,32 \
    --module-kwargs embed_dim=32 num_heads=4
```
