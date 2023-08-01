# Amazon Fine Food Review Analysis Project

## Introduction

This project focuses on sentiment analysis of Amazon Fine Food Reviews using Natural Language Processing (NLP) techniques. The primary goal is to build models to predict whether a review is negative or positive based on its text content. We will be using Bag of Words and N-Gram TFIDF approaches along with Random Forest, Logistic Regression, Naive Bayes, and XGBoost classifiers for the prediction.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Sentiment analysis is a valuable application of NLP, especially in the context of product reviews. Understanding customer sentiments can help businesses make data-driven decisions to improve their products and services. In this project, we analyze the Amazon Fine Food Reviews dataset to build machine learning models that can classify reviews as either negative or positive.Helps in making better recommendation based decisions for the existing and the new users by predicting review sentiment given by the customer. 

## Data

The dataset used for this project is the "Amazon Fine Food Reviews" dataset, which contains a large number of reviews and their corresponding sentiments. The dataset is available in CSV format and can be downloaded from [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews).

## Methodology

1. **Data Preprocessing**: Cleaning and preparing the text data, including removing special characters, converting text to lowercase, removing stopwords, and applying stemming/lemmatization and removing words doesn't exist in dictionary.

2. **Feature Extraction**: Creating the Bag of Words and N-Gram TFIDF representations of the reviews.

3. **Model Training**: Building and training the classifiers, including Random Forest, Logistic Regression, Naive Bayes, and XGBoost.

4. **Model Evaluation**: Evaluating the models' performance using the ROC AUC curve, which provides a comprehensive analysis of model performance for imbalanced datasets

## Results

The performance of each model is compared based on the ROC AUC curve, and the best-performing model for sentiment analysis is identified. Additionally, we analyze the impact of handling imbalanced data on model performance.

## Usage

To run the project, you need Python and the required dependencies installed on your system. You can follow the installation instructions below to set up the environment. After that, you can run the main script to perform sentiment analysis on new reviews.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- NLTK (Natural Language Toolkit)
- SQLite3

Clone Repository : git clone












