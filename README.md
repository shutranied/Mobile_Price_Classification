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
