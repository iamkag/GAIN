# GAIN - Generative Adversarial Imputation Networks

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

GAIN (Generative Adversarial Imputation Nets) is a model designed to address missing data imputation through adversarial training. This implementation, based on the paper by Jinsung Yoon, James Jordon, and Mihaela van der Schaar, applies generative adversarial networks to learn realistic values for missing entries in datasets, offering an advanced alternative to traditional imputation methods.

## Table of Contents

- [Overview](#overview)
- [Original Paper](#original-paper)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Traditional imputation techniques, such as mean or median imputation, can introduce bias and fail to capture the underlying data structure. GAIN offers a powerful solution by using a GAN framework to learn realistic values for missing data points, thus providing more robust and accurate imputed data. This model can be particularly valuable for preprocessing in machine learning workflows that require high-quality data.

## Original Paper

The concept of Generative Adversarial Imputation Nets (GAIN) is detailed in the following paper:
- **Title**: [GAIN: Missing Data Imputation using Generative Adversarial Nets](https://arxiv.org/abs/1806.02920)
- **Authors**: Jinsung Yoon, James Jordon, Mihaela van der Schaar
- **Publication Date**: June 8, 2018

## Features

- **Adversarial Imputation**: Imputes missing data using GAN-based techniques to better preserve realistic data distributions.
- **Customizable Hyperparameters**: Tune key parameters like batch size, hint rate, and alpha for optimal imputation results.
- **Easy Integration**: Flexible implementation that can be adapted to various datasets with missing values.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/iamkag/GAIN.git
   cd GAIN
   ```

2. **Install Dependencies**:
   It’s recommended to use a virtual environment.

   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

## Usage

The GAIN model is designed to be run on datasets containing missing values (represented as `NaN`). Below is an example of how to use GAIN for missing data imputation.

### Example Code

1. **Prepare Your Dataset**: Load your dataset, ensuring missing values are represented as `NaN`.
2. **Run the GAIN Model**:

   ```python
   import pandas as pd
   from gain import GAIN

   # Load data
   data = pd.read_csv('data_with_missing_values.csv')

   # Instantiate GAIN model
   gain = GAIN(data=data, batch_size=64, hint_rate=0.9, alpha=100)

   # Impute missing data
   imputed_data = gain.impute()

   # Save imputed data
   imputed_data.to_csv('imputed_data.csv', index=False)
   ```

   **Parameters**:
   - `data`: Dataset with missing values as `NaN`.
   - `batch_size`: Controls batch size for training.
   - `hint_rate`: The probability of providing hints for which values are missing.
   - `alpha`: Affects the weight of the reconstruction loss term.

### Results

Once the imputation process completes, `imputed_data` will contain the original dataset with missing values filled in. The imputed data can then be saved, analyzed, or used directly in machine learning workflows.

## Contributing

Contributions to enhance GAIN are encouraged! If you’d like to add features, fix issues, or improve performance, please:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License.

