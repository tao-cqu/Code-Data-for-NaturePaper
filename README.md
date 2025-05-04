# Dual-phase reservoir retention governs riverine microplastic fluxes
# Flux Analysis Model for Riverine Microplastics (FLUX-MP)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

> Machine learning models for predicting microplastic concentrations in river systems, including Random Forest, SVR, Neural Networks, and Ridge Regression.

## Table of Contents
- [Installation](#installation)
- [Data](#data-preparation)
- [Usage](#usage)
- [Auxiliary](#Auxiliary_time_recurrent_neural_network)

## Installation
### Conda Environment Setup
name: microplastics  
channels:
  - conda-forge      
  - defaults
dependencies:
  - python=3.8       
  - scikit-learn=1.3.0
  - pandas=1.4.3
  - numpy=1.21.5
  - matplotlib=3.5.2
  - tensorflow=2.10.0
  - pip:             
    - joblib==1.1.0
    - openpyxl==3.0.9
## Data
### Data
Supplementary table-References.xlsx

### Input Format
Features: 7 Dimensions for Non-buoyant MP , 5 Dimensions for Buoyant MP
Target: 1 Dimension

## Usage
### Training Models
Random Forest(selected)
Neural Network
SVR (RBF Kernel)
Ridge Regression

## Auxiliary time-recurrent neural network
Auxiliary_time_recurrent_neural_network_code.py

# Dam-Reservoir System MP Retention ratio
Matlab Code Dam-Reservoir System MP Retention ratio.txt
