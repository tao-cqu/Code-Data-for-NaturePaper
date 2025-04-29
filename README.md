# Code-Data-for-NaturePaper
# Flux Assessment Model for Riverine Microplastics (FAMRM)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

> Machine learning models for predicting microplastic concentrations in river systems, including Random Forest, SVR, Neural Networks, and Ridge Regression.

## Table of Contents
- [Installation](#installation)
- [Data](#data-preparation)
- [Usage](#usage)

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
### Input Format
Features: 7 Dimensions for Non-buoyant MP , 5 Dimensions for Buoyant MP
Target: 1 Dimension

## Usage
### Training Models
Random Forest
Neural Network
SVR (RBF Kernel)
Ridge Regression
