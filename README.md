# Mobile Price Classification - Code Decomposition Project

## 1. Project Overview

This project is based on a publicly available Kaggle notebook titled **Mobile Price Classification**.

Original Kaggle Notebook:  
https://www.kaggle.com/code/abdelruhmanessam/mobile-price-classification

The purpose of this project is to restructure the original notebook into a clean and organized Python project. The code has been decomposed into separate modules according to the main stages of a machine learning workflow, including data loading, data exploration, visualization, preprocessing, model training, and evaluation.

## 2. Dataset

The dataset used in this project is the Mobile Price Classification dataset. The main training file is:

- `train.csv`

The target variable is:

- `price_range`

The model predicts the price range category of a mobile phone based on features such as RAM, battery power, camera specifications, and connectivity features.

## 3. Project Folder Structure

```text
mobile_price_classification/
│── README.md
│── config.py
│── main.py
│── requirements.txt
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_loader.py
│
├── evaluation/
│   └── evaluation.py
│
├── models/
│   └── ml_models.py
│
├── preprocessing/
│   └── preprocessing.py
│
└── utils/
    ├── data_exploration.py
    └── visualization.py

```
## 4. Module Explanation

### 4.1 `config.py`

This file stores important project settings such as the dataset path, target column name, test size, random state, and model save path.

### 4.2 `data/data_loader.py`

This file contains the function for loading the dataset from a CSV file using pandas.

### 4.3 `utils/data_exploration.py`

This file contains functions for basic data exploration, including dataset information, descriptive statistics, unique values, missing values, and the first five rows.

### 4.4 `utils/visualization.py`

This file contains visualization functions such as class balance, RAM distribution, battery power distribution, and 3G support distribution.

### 4.5 `preprocessing/preprocessing.py`

This file handles preprocessing. It separates features and target variable, scales the features, and splits the dataset into training and testing sets.

### 4.6 `models/ml_models.py`

This file contains model training functions for Logistic Regression and Support Vector Classifier. It also includes functions to save and load the model.

### 4.7 `evaluation/evaluation.py`

This file contains model evaluation functions, including classification report and confusion matrix display.

### 4.8 `main.py`

This is the main file that runs the complete machine learning workflow from data loading until model evaluation and model saving.

## 5. How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## 6. Conclusion

This project shows how a Kaggle machine learning notebook can be decomposed into a structured Python project. The new structure makes the code easier to read, reuse, manage, and maintain.
