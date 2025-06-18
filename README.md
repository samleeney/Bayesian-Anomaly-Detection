# Bayesian Anomaly Detection

A demonstration of Bayesian anomaly detection for robust parameter inference in the presence of outliers.

## Overview

This repository demonstrates how Bayesian anomaly detection can automatically identify and mitigate the effects of outliers in scientific data analysis. The method uses a binary anomaly mask with Bernoulli priors and a piecewise likelihood that gracefully handles both normal and anomalous data points.

## Files

### 📓 [Jupyter Notebook](demo_anomaly_detection.ipynb)

### 🐍 [Python Script](bayesian_anomaly_detection.py)

## Quick Start

### Option 1: Jupyter Notebook
```bash
jupyter notebook demo_anomaly_detection.ipynb
```

### Option 2: Python Script
```bash
python bayesian_anomaly_detection.py
```

## Key References

- **Leeney et al. (2022)**: [Bayesian approach to radio frequency interference mitigation](https://arxiv.org/abs/2211.15448)
- **Anstey and Leeney (2023)**: [Enhanced Bayesian RFI Mitigation and Transient Flagging Using Likelihood Reweighting](https://arxiv.org/abs/2310.02146)
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
