# Bayesian Anomaly Detection

A demonstration of Bayesian anomaly detection for robust parameter inference in the presence of outliers.

## Overview

This repository demonstrates how Bayesian anomaly detection can automatically identify and mitigate the effects of outliers in scientific data analysis. The method uses a binary anomaly mask with Bernoulli priors and a piecewise likelihood that gracefully handles both normal and anomalous data points.

## Key Features

- **Automatic outlier detection**: No manual flagging required
- **Robust parameter inference**: Recovers true parameters despite contamination
- **Principled uncertainty quantification**: Proper Bayesian credible intervals
- **Computational efficiency**: Simple implementation requiring only a few lines of code modification
- **Flexible framework**: Works with any likelihood function

## Files

### 📓 [Jupyter Notebook](demo_anomaly_detection.ipynb)
Complete interactive demonstration with:
- Comprehensive mathematical framework with LaTeX equations
- Step-by-step implementation
- All visualisations embedded
- Detailed explanations and theory

### 🐍 [Python Script](bayesian_anomaly_detection.py)
Standalone Python script version:
- Command-line executable
- Same analysis as notebook
- Generates publication-quality plots
- Perfect for batch processing or integration into larger workflows

## Quick Start

### Option 1: Jupyter Notebook
```bash
jupyter notebook demo_anomaly_detection.ipynb
```

### Option 2: Python Script
```bash
python bayesian_anomaly_detection.py
```

## Method Overview

The approach introduces a **binary anomaly mask** ε_i for each data point:

```
ε_i = { 0 if data point i is expected
      { 1 if data point i is anomalous
```

Using Bernoulli priors and marginalising over all possible anomaly configurations leads to an **anomaly-corrected log-likelihood**:

```
log P(D|θ) = Σ max(log L_i + log(1-p), log p - log Δ)
```

This automatically flags outliers whilst preserving robust parameter estimation.

## Key References

- **Leeney et al. (2022)**: [Bayesian approach to radio frequency interference mitigation](https://arxiv.org/abs/2211.15448)
- **Anstey and Leeney (2023)**: [Enhanced Bayesian RFI Mitigation and Transient Flagging Using Likelihood Reweighting](https://arxiv.org/abs/2310.02146)

## Results

The demonstration shows dramatic improvements in parameter recovery:

| Parameter | True Value | Standard Method | Anomaly Corrected |
|-----------|------------|-----------------|-------------------|
| Slope (m) | 1.0        | Biased & Uncertain | 1.0 ± 0.1 |
| Intercept (c) | 1.0    | Severely Biased | 1.0 ± 1.0 |
| Noise (σ) | 5.0        | ~30 (6× too large) | 5.0 ± 1.0 |

## Dependencies

- NumPy
- Matplotlib
- SciPy (for optimisation in Python script)

## License

MIT License - see repository for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{leeney2022bayesian,
  title={Bayesian approach to radio frequency interference mitigation},
  author={Leeney, Samuel and others},
  journal={arXiv preprint arXiv:2211.15448},
  year={2022}
}
```