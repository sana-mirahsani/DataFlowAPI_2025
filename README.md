# DataFlowAPI_2025
Review binary classification by Logistic regression.

# Overview / Description
This project builds a review classifcation model for the french review using logistic regression.
<br>
It includes: data preprocessing, model training and evaluation. Plus, API and test API.

# Features
- Text preprocessing with custom cleaner
- Automated TF-IDF vectorization
- Connection with model by Fast API
- Testing API and model

# Project Structure

```
project_name/
│
├── api/                      # FastAPI endpoints
│   └── main.py
│
├── data/   # Training data
│   └── train.csv
│
├── models/      # Stored model and vocabulary         
│   ├── logistic_regression.pkl  # model
│   └── lr_vocab.pkl             # vocab
│
├── notebooks/  # Notebooks
│   └── training_nb.ipynb     # Training notebook
│
├── src/    # Utils
│   ├── data_utils.py         # data preparation functions
│   └── model_utils.py        # model training/prediction?evaluation               
│
├── tests/  # Testing API/Data
│   ├── test_api.py         # testing API
│   └── test_data_utils.py  # testing data utils
│
└── README.md                 # how to run, inputs/outputs, etc.

```

# Installation

## Clone the repository
git clone https://github.com/sana-mirahsani/DATAFLOWAPI_2025.git
cd DATAFLOWAPI_2025

## Install dependencies
pip install -r requirements.txt

# Usage
- Run **main.py**.

# Technologies Used
List key tools and libraries:

- Python 3.10

- Scikit-learn

- Pandas, NumPy 

- FastAPI

# Author
Sana Mirahsani
Master’s student in Machine Learning, University of Lille

Linkind : https://www.linkedin.com/in/sana-mirahsani

Github : https://github.com/sana-mirahsani 