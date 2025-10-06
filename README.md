# Lung Cancer Prediction

A web-based machine learning project for predicting lung cancer risk using Flask, Python, and MySQL. The project implements multiple ML algorithms such as Random Forest, Decision Tree, Logistic Regression, SVM, XGBoost, KNN, and Voting Classifier.

## Project Overview

This project allows users to:
- **Upload a dataset**: Train models using the dataset uploaded through a file upload page.
- **Manual Prediction**: Enter symptoms manually and predict the lung cancer risk (low or high).
- **Database Storage**: Input and output (prediction results) are stored in a MySQL database.
- **Performance Metrics**: View key performance metrics, including accuracy, confusion matrix, F1-score, recall, and precision.
  
The system includes a user interface with the following pages:
- **Home Page**: Includes buttons for navigating to the file upload, prediction, and performance pages.
- **File Upload Page**: Allows the user to upload data and train a model.
- **Manual Prediction Page**: Allows users to input symptoms and get predictions.
- **Performance Page**: Displays model performance metrics.
- **Sign In and Login Pages**: User authentication for secure access.

## Features

- **ML Models**: Random Forest, Decision Tree, Logistic Regression, SVM, XGBoost, KNN, Voting Classifier.
- **Data Storage**: All input and prediction data are stored in a MySQL database.
- **Performance Tracking**: Displays confusion matrix, F1-score, recall, precision, and accuracy of models.
- **Web Interface**: Built with HTML, CSS, and Flask for the backend, and MySQL for the database.

## Installation

To set up and run the project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/JesslaSelvaraj-03/Lung-Cancer-Prediction.git
