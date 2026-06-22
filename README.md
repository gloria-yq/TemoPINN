# TempoPINN

TempoPINN is a physics-informed learning framework for fixed-location temperature forecasting.  
It combines a State Predictor and a Dynamics Operator to model future temperature states and regularize their temporal evolution without requiring explicitly specified atmospheric equations.

This repository provides the core implementation of TempoPINN, including model definitions, configuration files, data loading utilities, and the main training entry.

## Overview

Physics-informed learning has shown effectiveness in weather
forecasting. Many existing methods rely on explicit governing equations
to enforce physical consistency. However, atmospheric variables in weather
systems are highly coupled, and their changes over time are difficult to
describe with explicit equations. To address this challenge, we propose
TempoPINN, a physics-informed framework for temperature forecasting at
fixed locations. We introduce State Predictor and Dynamics Operator as
two modules of TempoPINN. The former estimates future temperatures,
while the latter regularizes forecast changes without explicit atmospheric
equations. Both modules are implemented with a cross-time transformer.
Additionally, we introduce annual periodic consistency to capture seasonal
scale variations. Extensive experiments on WeatherBench temperature
time series show that TempoPINN outperforms representative neural
forecasting baselines in temperature forecasting.

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

python >= 3.8 \
torch\
numpy\
pandas\
scikit-learn\
matplotlib\
Install dependencies manually according to your local CUDA/PyTorch environment.

## Usage
Run TempoPINN with:\
python main.py \
  --data_path path/to/weather_data.csv \
  --model model/to/test\

