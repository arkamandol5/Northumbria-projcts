# Crime Prevention Analysis

This repository contains code and data for analyzing and predicting crime hotspots, performing causal inference, and predicting crime trends using various machine learning techniques.

## Project Structure

- `GBM_hotspot_prediction.ipynb`: Notebook for predicting crime hotspots using Gradient Boosting Machine (GBM).
- `bayesian_network_causal_inference.ipynb`: Notebook for performing causal inference using Bayesian Networks.
- `random_forest_causal_inference.ipynb`: Notebook for performing causal inference using Random Forest.
- `random_forest_trend_prediction.ipynb`: Notebook for predicting crime trends using Random Forest.
- `requirements.txt`: List of dependencies required to run the notebooks.
- `datasets/`: Directory containing datasets used in the analysis.
- `hotspots_map/`: Directory containing generated hotspot maps in HTML format.

## Datasets

The following datasets are used in the analysis:

- `UK_Police_Street_Crime_2018-10-01_to_2021-09-31.csv`: Crime data from the UK police (not included in the repository, can be downloaded from [Kaggle](https://www.kaggle.com/code/tantable/big-police-starter-notebook-19-million-crimes)).
- `Economically_inactive.xlsx`: Data on economically inactive population.
- `causal_inference_data_rf.csv`: Data for causal inference analysis using Random Forest.
- `education_levels.xlsx`: Data on education levels.
- `house_price_to_earning_ratio.xlsx`: Data on house price to earning ratio.
- `population-by-ethnicity-and-local-authority-2021.csv`: Population data by ethnicity and local authority.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
