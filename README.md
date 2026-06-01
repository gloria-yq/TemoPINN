# TempoPINN

TempoPINN is a physics-informed learning framework for fixed-location temperature forecasting.  
It combines a State Predictor and a Dynamics Operator to model future temperature states and regularize their temporal evolution without requiring explicitly specified atmospheric equations.

This repository provides the core implementation of TempoPINN, including model definitions, configuration files, data loading utilities, and the main training entry.

## Overview

TempoPINN is designed for multivariate weather time series forecasting at a fixed spatial location. Given a historical observation window, the model predicts future 2-meter temperature at a specified forecast horizon.

The framework contains three main components:

- **State Predictor**: estimates the future temperature conditioned on the forecast horizon.
- **Dynamics Operator**: learns an implicit temporal evolution relation and regularizes forecast changes.
- **Annual Periodic Consistency**: introduces seasonal-scale regularization for meteorological time series.

Both the State Predictor and Dynamics Operator are implemented with a Cross-Time Transformer module, which combines temporal self-attention with cross-time feed-forward transformation.

## Repository Structure

```text
TempoPINN/
├── configs/          # Configuration and argument parser
├── dataloader/       # Weather data preprocessing and dataloader
├── Model/            # Model definitions
├── utils/            # Utility functions
├── main.py           # Main training and evaluation entry
└── README.md

```
## Requirements
The code is implemented in Python and PyTorch.

Recommended environment:

python >= 3.8
torch
numpy
pandas
scikit-learn
matplotlib
Install dependencies manually according to your local CUDA/PyTorch environment.
